#!/usr/bin/env python3

import argparse
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


def prepare_features(data, columns):
    ranks = data.groupby('QueryId')[columns].rank(pct=True)
    ranks.columns = [f'query_rank_{i}' for i in range(len(columns))]
    return pd.concat([data[columns], ranks], axis=1).astype('float32')


def ndcg(data, scores):
    labels = data['Label'].to_numpy()
    queries = data['QueryId'].to_numpy()
    starts = np.r_[0, np.flatnonzero(queries[1:] != queries[:-1]) + 1, len(queries)]
    result = []

    for left, right in zip(starts[:-1], starts[1:]):
        gains = 2.0 ** labels[left:right] - 1
        k = min(10, len(gains))
        weights = 1 / np.log2(np.arange(2, k + 2))
        ideal = np.sort(gains)[::-1][:k] @ weights
        if ideal:
            order = np.argsort(-scores[left:right], kind='stable')[:k]
            result.append((gains[order] @ weights) / ideal)
        else:
            result.append(0.0)

    return float(np.mean(result))


def train(data_dir, model_file):
    data = pd.read_csv(Path(data_dir) / 'train.csv')
    data = data.sort_values('QueryId').reset_index(drop=True)
    columns = [c for c in data.columns if c not in ('QueryId', 'Label')]
    features = prepare_features(data, columns)

    queries = data['QueryId'].unique().copy()
    np.random.default_rng(42).shuffle(queries)
    valid = data['QueryId'].isin(queries[:max(1, len(queries) // 5)])

    train_pool = Pool(features.loc[~valid], data.loc[~valid, 'Label'], group_id=data.loc[~valid, 'QueryId'])
    valid_pool = Pool(features.loc[valid], data.loc[valid, 'Label'], group_id=data.loc[valid, 'QueryId'])

    params = dict(
        loss_function='YetiRank',
        eval_metric='NDCG:top=10',
        iterations=700,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=80)

    best_iterations = model.best_iteration_ + 1
    score = ndcg(data.loc[valid], model.predict(features.loc[valid]))
    print(f'validation nDCG@10: {score:.5f}, iterations: {best_iterations}')

    params['iterations'] = best_iterations
    model = CatBoostRanker(**params)
    model.fit(Pool(features, data['Label'], group_id=data['QueryId']))
    model.save_model(str(model_file))


def predict(data_dir, model_file, submission_file):
    data = pd.read_csv(Path(data_dir) / 'test.csv')
    columns = [c for c in data.columns if c not in ('ObjectId', 'QueryId')]
    features = prepare_features(data, columns)

    model = CatBoostRanker()
    model.load_model(str(model_file))
    scores = model.predict(features[model.feature_names_])
    pd.DataFrame({'ObjectId': data['ObjectId'], 'Score': scores}).to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--submission_file')
    parser.add_argument('data_dir')
    args = parser.parse_args()

    start = timer()
    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if not args.submission_file:
            parser.error('нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)
    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
