#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
from collections import Counter
import logging
import argparse
import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import data_helper
from common import common_function

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# TODO: set %env JOBLIB_TEMP_FOLDER=/tmp, otherwise will raise "OSError: [Errno 28] No space left on device"
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--train_data_file", type=str, default='/ceph/qbkg2/aitingliu/qq/data/20190726/raw/train_check.txt')
parser.add_argument("--valid_data_file", type=str, default='/ceph/qbkg2/aitingliu/qq/data/20190726/raw/dev.txt')
parser.add_argument("--test_data_file", type=str, default='/ceph/qbkg2/aitingliu/qq/data/20190726/raw/test_check.txt')
parser.add_argument("--x_train_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/x_train.csv')
parser.add_argument("--x_valid_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/x_valid.csv')
parser.add_argument("--x_test_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/x_test.csv')
parser.add_argument("--y_train_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/y_train.csv')
parser.add_argument("--y_valid_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/y_valid.csv')
parser.add_argument("--y_test_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/y_test.csv')
parser.add_argument("--dtrain_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/dtrain.buffer')
parser.add_argument("--dvalid_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/dvalid.buffer')
parser.add_argument("--dtest_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/dtest.buffer')
parser.add_argument("--pred_data_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/results_xgb.csv')
parser.add_argument("--word_counts_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/word_counts')
parser.add_argument("--feature_map_file", type=str,
                    default='/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/feature_map')
parser.add_argument("--train_scwlstm_pred_file", type=str,
                    default="/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM/best_eval_loss/output_train")
parser.add_argument("--valid_scwlstm_pred_file", type=str,
                    default="/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM/best_eval_loss/output_dev")
parser.add_argument("--test_scwlstm_pred_file", type=str,
                    default="/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM/best_eval_loss/output_test")
parser.add_argument("--use_scwlstm", dest='use_scwlstm', action='store_true')
parser.add_argument("--no_use_scwlstm", dest='use_scwlstm', action='store_false')
parser.add_argument("--model_path", type=str, default="/ceph/qbkg2/aitingliu/qq/XGBoost/model/with_scwlstm/model")
parser.add_argument("--embedding_dim", type=int, default=200)
parser.add_argument("--embedding_file", type=str, default='/ceph/qbkg2/winsechang/MODEL/qq_simscore/wdic/word2vec.dict')
parser.add_argument("--stopword_file", type=str, default='/ceph/qbkg2/winsechang/MODEL/qq_simscore/wdic/stopwords.txt')
parser.add_argument("--early_stopping_rounds", type=int, default=50)
parser.add_argument("--num_boost_round", type=int, default=200)
parser.add_argument("--booster", type=str, default="gbtree", help="[default= gbtree ]")
parser.add_argument("--eta", type=float, default=0.3, help="[default=0.3]")
parser.add_argument("--gamma", type=float, default=0, help="[default=0]")
parser.add_argument("--max_depth", type=float, default=6, help="[default=6]")
parser.add_argument("--min_child_weight", type=float, default=1, help="[default=1]")
parser.add_argument("--max_delta_step", type=float, default=0, help="[default=0]")
parser.add_argument("--subsample", type=float, default=1, help="[default=1]")
parser.add_argument("--colsample_bytree", type=float, default=1, help="[default=1]")
parser.add_argument("--colsample_bylevel", type=float, default=1, help="[default=1]")
parser.add_argument("--lamda", type=float, default=1, help="[default=1]")
parser.add_argument("--alpha", type=float, default=0, help="[default=0]")
parser.add_argument("--scale_pos_weight", type=float, default=1, help="[default=1]")
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--eval_metric", type=str, default="error,logloss")
args = parser.parse_args()
"""
https://xgboost.readthedocs.io/en/latest/parameter.html
"""
common_function.print_args(args)

common_function.makedir(args.pred_data_file)
common_function.makedir(args.word_counts_file)
common_function.makedir(args.stopword_file)
common_function.makedir(args.model_path)


def first_train():
    # TODO: Saving DMatrix into a XGBoost binary file will make loading faster
    #######################################################################

    logger.info("Starting to read Embedding file...")
    word2vec = common_function.load_word2vec(args.embedding_file, filter_num=args.embedding_dim)
    logger.info("Finish reading Embedding file !")
    logger.info('Found %d word vectors of word2vec' % len(word2vec))

    stop_words = common_function.load_file_2_dict(args.stopword_file, colum=1)
    logger.info("Finish reading stopword file !")
    logger.info('Stopword is : ' + "|".join(list(stop_words.keys())))

    ############################## reading data  #############################
    logger.info("Starting to read training samples...")
    train_texts_1, train_texts_2, labels = data_helper.read_data(args.train_data_file)
    val_texts_1, val_texts_2, val_labels = data_helper.read_data(args.valid_data_file)
    test_texts_1, test_texts_2, test_labels = data_helper.read_data(args.test_data_file)
    logger.info("Finish reading training samples !")

    train_orig = pd.DataFrame({"question1": train_texts_1, "question2": train_texts_2})
    val_orig = pd.DataFrame({"question1": val_texts_1, "question2": val_texts_2})
    test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})

    ############################ save words counts ############################
    total_words = []
    ques = pd.concat([train_orig, val_orig], axis=0).reset_index(drop='index')
    for i in range(ques.shape[0]):
        total_words += ques.question1[i].split('|')
        total_words += ques.question2[i].split('|')

    ## save word freq to total_counts
    counts = Counter(total_words)
    r = open(args.word_counts_file, 'w', encoding="utf-8")
    for _word, _count in counts.items():
        r.write("%s\t%d\n" % (_word, _count))
    r.close()

    ############################## basic features  #############################
    train_cp = train_orig.copy()
    val_cp = val_orig.copy()
    test_cp = test_orig.copy()
    x_train_basic = data_helper.get_basic_feat(train_cp, args.embedding_dim, stop_words, word2vec)
    x_valid_basic = data_helper.get_basic_feat(val_cp, args.embedding_dim, stop_words, word2vec)
    x_test_basic = data_helper.get_basic_feat(test_cp, args.embedding_dim, stop_words, word2vec)

    ####################### sentence word char features #########################
    weights = {word: data_helper.get_weight(count) for word, count in counts.items()}
    x_train_more = data_helper.build_features(train_orig, stop_words, weights)
    x_valid_more = data_helper.build_features(val_orig, stop_words, weights)
    x_test_more = data_helper.build_features(test_orig, stop_words, weights)

    ######################## SCWLSTM model simscore #############################
    if args.use_scwlstm:
        x_train_sim = data_helper.model_simscore(args.train_scwlstm_pred_file, train_cp)
        x_valid_sim = data_helper.model_simscore(args.valid_scwlstm_pred_file, val_cp)
        x_test_sim = data_helper.model_simscore(args.test_scwlstm_pred_file, test_cp)

        ################### combine all features ##############################
        x_train = pd.concat((x_train_basic, x_train_more, x_train_sim), axis=1)
        x_valid = pd.concat((x_valid_basic, x_valid_more, x_valid_sim), axis=1)
        x_test = pd.concat((x_test_basic, x_test_more, x_test_sim), axis=1)
    else:
        x_train = pd.concat((x_train_basic, x_train_more), axis=1)
        x_valid = pd.concat((x_valid_basic, x_valid_more), axis=1)
        x_test = pd.concat((x_test_basic, x_test_more), axis=1)

    x_train.drop(['question1', 'question2'], axis=1, inplace=True)
    x_valid.drop(['question1', 'question2'], axis=1, inplace=True)
    x_test.drop(['question1', 'question2'], axis=1, inplace=True)

    # print(x_train.columns)

    features = [x for x in x_train.columns]
    data_helper.ceate_feature_map(args.feature_map_file, features)

    x_train.columns = [str(i) for i in range(x_train.shape[1])]
    x_valid.columns = [str(i) for i in range(x_valid.shape[1])]
    x_test.columns = [str(i) for i in range(x_test.shape[1])]

    ################################ save csv ###############################
    logger.info("Start saving csv.")
    x_train.to_csv(args.x_train_file, header=False, index=False, encoding="utf-8", sep="\t")
    x_valid.to_csv(args.x_valid_file, header=False, index=False, encoding="utf-8", sep="\t")
    x_test.to_csv(args.x_test_file, header=False, index=False, encoding="utf-8", sep="\t")

    y_train = pd.DataFrame(labels)
    y_valid = pd.DataFrame(val_labels)
    y_test = pd.DataFrame(test_labels)
    y_train.to_csv(args.y_train_file, header=False, index=False, encoding="utf-8", sep="\t")
    y_valid.to_csv(args.y_valid_file, header=False, index=False, encoding="utf-8", sep="\t")
    y_test.to_csv(args.y_test_file, header=False, index=False, encoding="utf-8", sep="\t")
    logger.info("Done saving csv.")

    ############################# save DMatrix ################################
    logger.info("Start saving DMatrix.")
    y_train = labels
    y_valid = val_labels
    y_test = test_labels
    d_train = xgb.DMatrix(x_train, label=labels)
    d_valid = xgb.DMatrix(x_valid, label=val_labels)
    d_test = xgb.DMatrix(x_test, label=test_labels)
    d_train.save_binary(args.dtrain_file, silent=False)
    d_valid.save_binary(args.dvalid_file, silent=False)
    d_test.save_binary(args.dtest_file, silent=False)
    # logger.info(d_train.feature_names)
    logger.info("Done saving DMatrix.")

    ############################# train models #################################
    params = {
        "booster": args.booster,
        "eta": args.eta,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_delta_step": args.max_delta_step,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "lambda": args.lamda,
        "alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
        "eval_metric": list(args.eval_metric.split(","))
    }

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, args.num_boost_round, watchlist, early_stopping_rounds=args.early_stopping_rounds)
    bst.save_model(args.model_path)
    bst.dump_model(args.model_path + '.dump')

    ## make the submission
    p_test = bst.predict(xgb.DMatrix(x_test))
    df_sub = pd.DataFrame(
        {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
    df_sub.to_csv(args.pred_data_file,
                  header=False, index=False, encoding='utf-8', sep="\t",
                  columns=['user_query', 'candidate_query', 'label', 'score'])

    ## make the submission for best
    p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=bst.best_ntree_limit)
    df_sub = pd.DataFrame(
        {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
    df_sub.to_csv(args.pred_data_file + "_best",
                  header=False, index=False, encoding='utf-8', sep="\t",
                  columns=['user_query', 'candidate_query', 'label', 'score'])

    logger.info("best_iteration: {}".format(bst.best_iteration))
    logger.info("ntree_limit=bst.best_ntree_limit: {}".format(bst.best_ntree_limit))
    logger.info("best_score: {}".format(bst.best_score))


def train():
    # TODO: Saving DMatrix into a XGBoost binary file will make loading faster
    ################################################################
    ######################### reading data  ########################
    logger.info("Starting to read training samples...")
    train_texts_1, train_texts_2, labels = data_helper.read_data(args.train_data_file)
    val_texts_1, val_texts_2, val_labels = data_helper.read_data(args.valid_data_file)
    test_texts_1, test_texts_2, test_labels = data_helper.read_data(args.test_data_file)
    logger.info("Finish reading training samples !")

    ######################### load csv data ########################
    # # TODO: load a CSV file into DMatrix
    logger.info("Start loading csv.")
    x_train = pd.read_csv(args.x_train_file, header=None, encoding="utf-8", sep="\t")
    x_valid = pd.read_csv(args.x_valid_file, header=None, encoding="utf-8", sep="\t")
    x_test = pd.read_csv(args.x_test_file, header=None, encoding="utf-8", sep="\t")

    y_train = pd.read_csv(args.y_train_file, header=None, encoding="utf-8", sep="\t")
    y_valid = pd.read_csv(args.y_valid_file, header=None, encoding="utf-8", sep="\t")
    # y_test = pd.read_csv(args.y_test_file, header=None, encoding="utf-8", sep="\t")

    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    logger.info("Done loading csv.")

    ########################## load DMatrix #########################
    # # TODO: load a XGBoost binary file into DMatrix
    # d_train = xgb.DMatrix(args.dtrain_file)
    # d_valid = xgb.DMatrix(args.dvalid_file)
    # d_test = xgb.DMatrix(args.dtest_file)

    ########################### train models ########################
    params = {
        "booster": args.booster,
        "eta": args.eta,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_delta_step": args.max_delta_step,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "lambda": args.lamda,
        "alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
        "eval_metric": list(args.eval_metric.split(","))
    }

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, args.num_boost_round, watchlist, early_stopping_rounds=args.early_stopping_rounds)
    bst.save_model(args.model_path)
    bst.dump_model(args.model_path + '.dump')

    ## make the submission
    p_test = bst.predict(xgb.DMatrix(x_test))
    df_sub = pd.DataFrame(
        {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
    df_sub.to_csv(args.pred_data_file,
                  header=False, index=False, encoding='utf-8', sep="\t",
                  columns=['user_query', 'candidate_query', 'label', 'score'])

    ## make the submission for best
    p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=bst.best_ntree_limit)
    df_sub = pd.DataFrame(
        {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
    df_sub.to_csv(args.pred_data_file + "_best",
                  header=False, index=False, encoding='utf-8', sep="\t",
                  columns=['user_query', 'candidate_query', 'label', 'score'])

    logger.info("best_iteration: {}".format(bst.best_iteration))
    logger.info("ntree_limit=bst.best_ntree_limit: {}".format(bst.best_ntree_limit))
    logger.info("best_score: {}".format(bst.best_score))


def grid_search_params():
    # TODO: load a CSV file into DMatrix
    logger.info("Start loading csv.")
    x_train = pd.read_csv(args.x_train_file, header=None, encoding="utf-8", sep="\t")
    y_train = pd.read_csv(args.y_train_file, header=None, encoding="utf-8", sep="\t")
    logger.info("Done loading csv.")

    ######################## grid seach params ############################
    other_params = {
        "booster": args.booster,
        # "learning_rate": args.eta,  # [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        # "n_estimators": args.num_boost_round,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_delta_step": args.max_delta_step,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "reg_lambda": args.lamda,
        "reg_alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
        "eval_metric": list(args.eval_metric.split(","))
    }
    cv_params = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2]
    }
    model = xgb.XGBClassifier(**other_params)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1213)
    grid_search = GridSearchCV(model, param_grid=cv_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(x_train, y_train[0])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    # first_train()
    # train()
    grid_search_params()
