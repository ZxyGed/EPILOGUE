import os
import argparse
import time
import numpy as np
import cupy
import cuml
from cuml.svm import SVC
from cuml.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier # can't use cuml.multiclass.OneVsRestClassifier, it will raise error on estimator__C
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, zero_one_loss
from metrics import evaluate_performance, accuracy_top


def parseArgs4Evaluate():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('-d', '--dataset', default='yeast', type=str)
    parser.add_argument('-sbi', '--subsetInfo', default='level1', type=str)
    
    # parser.add_argument('-ebf', '--embeddingFile', default='autodl-tmp/yeast_emb_512_3200_3_2_0.4_seqFeature.npy', type=str)
    # parser.add_argument('-msg', '--message', default='yeast_emb_512_3200_3_2_0.4_seqFeature', type=str)
    parser.add_argument('-gi', '--gpu_idx', type=int, default=0)
    parser.add_argument('-ebf', '--embeddingFile', default='autodl-tmp/yeast_emb_256_6400_2_1_0.4_identity.npy', type=str)
    parser.add_argument('-msg', '--message', default='yeast_emb_256_6400_2_1_0.4_identity', type=str)
    return parser.parse_known_args()

def evaluate_cmd():
    t = time.time()
    args, unknown = parseArgs4Evaluate()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
    embedding = np.load(args.embeddingFile)
    config = {'gamma': {'minimum': -3, 'maximum': 0, 'samples': 4}, 'regularization': {'minimum': -2, 'maximum': 2, 'samples': 5}, 'trials': 1, 'test_size': 0.2, 'folds': 3}
    evaluate_func(embedding, args.dataset, args.subsetInfo, args.message, config, args.gpu_idx)
    print("Total time elapsed: %.4fs" % (time.time() - t))


def evaluate_func(embedding, dataset, subsetInfo, message, config, gpu_idx=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    anno = np.load(f'autodl-tmp/data/labels/{dataset}_{subsetInfo}_label.npy').T
    anno = anno[:,anno.sum(axis=0)>=10]
    idx = np.sum(anno,axis=1)!=0
    
    idx1 = np.load('autodl-tmp/yeast_bionic_idx.npy')
    idx = idx*idx1
    
    dataset, standard = embedding[idx].astype(np.float32), anno[idx].astype(np.float32)
    scores, f1_all = _evaluate(dataset, standard, config, gpu_idx)
    with open('log.txt', 'a') as f:
        f.write(f'{scores} {message} {subsetInfo}\n')
        f.write(f'{f1_all} {message} {subsetInfo}\n')

def _evaluate(dataset, standard, config, gpu_idx):
    cupy.cuda.Device(gpu_idx).use()
    if isinstance(dataset, np.ndarray):
        dataset = cupy.asarray(dataset)
        standard = cupy.asarray(standard)
    elif isinstance(dataset, torch.Tensor):
        from torch.utils.dlpack import to_dlpack
        from cupy import fromDlpack
        dataset = fromDlpack(to_dlpack(dataset.cuda()))
        standard = fromDlpack(to_dlpack(standard.cuda()))
        
    # preprocess
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    
    gamma_config = config["gamma"]
    reg_config = config["regularization"]  # regularization config
    gamma = np.exp(np.linspace(gamma_config["minimum"], gamma_config["maximum"], gamma_config["samples"]))
    regularization = np.exp(np.linspace(reg_config["minimum"], reg_config["maximum"], reg_config["samples"]))

    # track scores
    scores = []
    f1_all = []

    for trial in range(config["trials"]):
        # Get train-test split. Random state ensures all datasets get the
        # same splits so comparison is fair.
        X_train, X_test, y_train, y_test = train_test_split(
            dataset,
            standard,
            test_size=config["test_size"],
            random_state=42,
        )

        estimator = OneVsRestClassifier(SVC(cache_size=300, probability=True))

        parameters = {"estimator__gamma": gamma, "estimator__C": regularization}
        
        model = GridSearchCV(
            estimator,
            param_grid=parameters,
            cv=min(X_train.shape[0], config["folds"]),
            n_jobs=1, # useGPU, shoude be set to 1
            verbose=1,
            scoring="f1_micro",
            refit="f1_micro",
        )
        # print(model.cv)
        
        model.fit(X_train.get(), y_train.get())
        svc_list = model.best_estimator_.estimators_
        y_prob_list = [s.predict_proba(X_test)[:, 1].reshape(-1, 1) for s in svc_list]
        y_prob_list = [s.decision_function(X_test).reshape(-1, 1) for s in svc_list]
        y_prob = np.hstack(y_prob_list)
        # print(y_prob.get().min())
        y_prob = (y_prob+1)/2
        y_pred = model.predict(X_test.get())
        
        y_test = y_test.get()
        y_prob = y_prob.get()

        acc = f1_score(y_test, y_pred, average="samples")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        micro_f1 = f1_score(y_test, y_pred, average="micro")
        zero = zero_one_loss(y_test, y_pred)
        acc_top=accuracy_top(y_pred, y_test)

        result = evaluate_performance(y_test, y_prob, y_pred)
        acc_top = result['acc']
        f1 = result['F1']
        m_aupr = result["m-aupr"]
        M_aupr = result["M-aupr"]
        zero = result['zero']

        scores.append([acc, macro_f1, micro_f1, acc_top, f1, m_aupr, M_aupr, zero])
        # scores.append([acc, macro_f1, micro_f1, zero])
        # f1_all.append(f1_score(y_test, y_pred, average=None))
    # return scores, f1_all
    return scores


if __name__=='__main__':
    evaluate_cmd()