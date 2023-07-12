#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import os
import datetime
from collections import defaultdict

from joblib import parallel_backend


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import scanpy as sc

from sklearn.metrics import RocCurveDisplay

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics
import umap

from imblearn.ensemble import BalancedRandomForestClassifier


# import torch
import tensorflow as tf
from tensorflow import keras
import harmonypy as hm

# from src.da_models.adda import ADDAST
# from src.da_models.dann import DANN
# from src.da_models.datasets import SpotDataset
# from src.utils.evaluation import JSD
from src.utils.data_loading import load_spatial, load_sc, get_selected_dir

# datetime object containing current date and time
script_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")


# %%
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# if device == "cpu":
#     warnings.warn("Using CPU", stacklevel=2)


TRAIN_USING_ALL_ST_SAMPLES = False
N_MARKERS = 20
ALL_GENES = False

ST_SPLIT = False
N_SPOTS = 20000
N_MIX = 8
SCALER_NAME = "minmax"

SAMPLE_ID_N = "151673"

BATCH_SIZE = 512
NUM_WORKERS = 16


DATA_DIR = "../AGrEDA/data"
# DATA_DIR = "./data_combine"

ST_SPLIT = False

MODEL_NAME = "CellDART"
MODEL_VERSION = "V1"
PRETRAINING = True

MILISI = True

# %%
# results_folder = os.path.join("results", MODEL_NAME, script_start_time)
# model_folder = os.path.join("model", MODEL_NAME, script_start_time)

model_folder = os.path.join("model", MODEL_NAME, MODEL_VERSION)
results_folder = os.path.join("results", MODEL_NAME, MODEL_VERSION)


if not os.path.isdir(results_folder):
    os.makedirs(results_folder)


# %%
# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


# %% [markdown]
#   # Data load


# %%
print("Loading Data")
# Load spatial data

selected_dir = get_selected_dir(DATA_DIR, N_MARKERS, ALL_GENES)
# Load spatial data
mat_sp_d, mat_sp_train, st_sample_id_l = load_spatial(
    selected_dir,
    SCALER_NAME,
    train_using_all_st_samples=TRAIN_USING_ALL_ST_SAMPLES,
    st_split=ST_SPLIT,
)

# Load sc data
sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2 = load_sc(
    selected_dir,
    SCALER_NAME,
    n_mix=N_MIX,
    n_spots=N_SPOTS,
)


# %% [markdown]
#   # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#   ## Prepare dataloaders

# %% [markdown]
#   ## Define Model

# %%
pretrain_folder = os.path.join(model_folder, "pretrain")
advtrain_folder = os.path.join(model_folder, "advtrain")


# %%
# st_sample_id_l = [SAMPLE_ID_N]


# %% [markdown]
#  ## Load Models

# %%
# checkpoints_da_d = {}
print("Getting predictions: ")
pred_sp_d = {}

if TRAIN_USING_ALL_ST_SAMPLES:
    model = keras.models.load_model(
        os.path.join(advtrain_folder, "all_st_samps", "final_model")
    )
    for sample_id in st_sample_id_l:
        pred_sp_d[sample_id] = model.predict(mat_sp_d[sample_id]["test"])

    # TODO: add pretraining
    if PRETRAINING:
        pred_sp_noda_d = {}
        model_noda = keras.models.load_model(
            os.path.join(pretrain_folder, "all_st_samps", "final_model")
        )
        for sample_id in st_sample_id_l:
            pred_sp_noda_d[sample_id] = model_noda.predict(mat_sp_d[sample_id]["test"])

else:
    for sample_id in st_sample_id_l:
        model = keras.models.load_model(
            os.path.join(advtrain_folder, sample_id, "final_model")
        )
        pred_sp_d[sample_id] = model.predict(mat_sp_d[sample_id]["test"])

    if PRETRAINING:
        pred_sp_noda_d = {}
        for sample_id in st_sample_id_l:
            model_noda = keras.models.load_model(
                os.path.join(pretrain_folder, sample_id, "final_model")
            )
            pred_sp_noda_d[sample_id] = model_noda.predict(mat_sp_d[sample_id]["test"])


# %% [markdown]
#  ## Evaluation of latent space

# %%
rf50_d = {"da": {}, "noda": {}}
splits = ["train", "val", "test"]
for split in splits:
    for k in rf50_d:
        rf50_d[k][split] = {}

if MILISI:
    miLISI_d = {"da": {}}
    if PRETRAINING:
        miLISI_d["noda"] = {}
    splits = ["train", "val", "test"]
    for split in splits:
        for k in miLISI_d:
            miLISI_d[k][split] = {}

Xs = [sc_mix_d["train"], sc_mix_d["val"], sc_mix_d["test"]]
random_states = np.asarray([225, 53, 92])

for sample_id in st_sample_id_l:
    print(f"Calculating domain shift for {sample_id}:", end=" ")
    random_states = random_states + 1

    # model = keras.models.load_model(
    #     os.path.join(advtrain_folder, sample_id, "final_model")
    # )
    embs = keras.models.load_model(os.path.join(advtrain_folder, sample_id, "embs"))
    if PRETRAINING:
        # model_noda = keras.models.load_model(
        #     os.path.join(pretrain_folder, sample_id, "final_model")
        # )
        embs_noda = keras.models.load_model(
            os.path.join(pretrain_folder, sample_id, "embs")
        )

    for i, (split, X, rs) in enumerate(zip(splits, Xs, random_states)):
        print(split, end=" ")
        figs = []

        X_target = mat_sp_d[sample_id]["test"]

        y_dis = np.concatenate(
            [
                np.zeros((X.shape[0],), dtype=np.int_),
                np.ones((X_target.shape[0],), dtype=np.int_),
            ]
        )

        source_emb = embs.predict(X)

        target_emb = embs.predict(X_target)

        emb = np.concatenate([source_emb, target_emb])

        if PRETRAINING:
            source_emb_noda = embs_noda.predict(X)
            target_emb_noda = embs_noda.predict(X_target)

            emb_noda = np.concatenate([source_emb_noda, target_emb_noda])

        n_cols = 2 if PRETRAINING else 1
        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 10, 10), squeeze=False)
        pca_da_df = pd.DataFrame(
            PCA(n_components=2).fit_transform(emb), columns=["PC1", "PC2"]
        )

        pca_da_df["domain"] = ["source" if x == 0 else "target" for x in y_dis]
        sns.scatterplot(
            data=pca_da_df, x="PC1", y="PC2", hue="domain", ax=axs[0][0], marker="."
        )

        if PRETRAINING:
            pca_noda_df = pd.DataFrame(
                PCA(n_components=2).fit_transform(emb_noda), columns=["PC1", "PC2"]
            )
            pca_noda_df["domain"] = pca_da_df["domain"]

            sns.scatterplot(
                data=pca_noda_df,
                x="PC1",
                y="PC2",
                hue="domain",
                ax=axs[0][1],
                marker=".",
            )

        for ax in axs.flat:
            ax.set_aspect("equal", "box")
        fig.suptitle(f"{sample_id} {split}")
        fig.savefig(
            os.path.join(results_folder, f"PCA_{sample_id}_{split}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        with parallel_backend("threading", n_jobs=-1):
            if MILISI:
                meta_df = pd.DataFrame(y_dis, columns=["Domain"])
                miLISI_d["da"][split][sample_id] = np.median(
                    hm.compute_lisi(emb, meta_df, ["Domain"])
                )

                if PRETRAINING:
                    miLISI_d["noda"][split][sample_id] = np.median(
                        hm.compute_lisi(emb_noda, meta_df, ["Domain"])
                    )

        if PRETRAINING:
            (
                emb_train,
                emb_test,
                emb_noda_train,
                emb_noda_test,
                y_dis_train,
                y_dis_test,
            ) = model_selection.train_test_split(
                emb,
                emb_noda,
                y_dis,
                test_size=0.2,
                random_state=rs,
                stratify=y_dis,
            )
        else:
            (
                emb_train,
                emb_test,
                y_dis_train,
                y_dis_test,
            ) = model_selection.train_test_split(
                emb, y_dis, test_size=0.2, random_state=rs, stratify=y_dis
            )

        pca = PCA(n_components=50)
        emb_train_50 = pca.fit_transform(emb_train)
        emb_test_50 = pca.transform(emb_test)

        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=-1)
        clf.fit(emb_train_50, y_dis_train)
        y_pred_test = clf.predict(emb_test_50)

        # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
        bal_accu_test = metrics.balanced_accuracy_score(y_dis_test, y_pred_test)

        rf50_d["da"][split][sample_id] = bal_accu_test

        if PRETRAINING:
            pca = PCA(n_components=50)
            emb_noda_train_50 = pca.fit_transform(emb_noda_train)
            emb_noda_test_50 = pca.transform(emb_noda_test)

            clf = BalancedRandomForestClassifier(random_state=145, n_jobs=-1)
            clf.fit(emb_noda_train_50, y_dis_train)
            y_pred_noda_test = clf.predict(emb_noda_test_50)

            # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
            bal_accu_noda_test = metrics.balanced_accuracy_score(
                y_dis_test, y_pred_noda_test
            )

            rf50_d["noda"][split][sample_id] = bal_accu_noda_test
    # newline at end of split
    print("")


# %% [markdown]
#   # 4. Predict cell fraction of spots and visualization

# %%
adata_spatialLIBD = sc.read_h5ad(os.path.join(selected_dir, "adata_spatialLIBD.h5ad"))

adata_spatialLIBD_d = {}
for sample_id in st_sample_id_l:
    adata_spatialLIBD_d[sample_id] = adata_spatialLIBD[
        adata_spatialLIBD.obs.sample_id == sample_id
    ]
    adata_spatialLIBD_d[sample_id].obsm["spatial"] = (
        adata_spatialLIBD_d[sample_id].obs[["X", "Y"]].values
    )


# %%
num_name_exN_l = []
for k, v in sc_sub_dict.items():
    if "Ex" in v:
        num_name_exN_l.append((k, v, int(v.split("_")[1])))
num_name_exN_l.sort(key=lambda a: a[2])
num_name_exN_l


# %%
Ex_to_L_d = {
    1: {5, 6},
    2: {5},
    3: {4, 5},
    4: {6},
    5: {5},
    6: {4, 5, 6},
    7: {4, 5, 6},
    8: {5, 6},
    9: {5, 6},
    10: {2, 3, 4},
}


# %%
numlist = [t[0] for t in num_name_exN_l]
Ex_l = [t[2] for t in num_name_exN_l]
num_to_ex_d = dict(zip(numlist, Ex_l))


# %%
def plot_cellfraction(visnum, adata, pred_sp, ax=None):
    """Plot predicted cell fraction for a given visnum"""
    adata.obs["Pred_label"] = pred_sp[:, visnum]
    # vmin = 0
    # vmax = np.amax(pred_sp)

    sc.pl.spatial(
        adata,
        img_key="hires",
        color="Pred_label",
        palette="Set1",
        size=1.5,
        legend_loc=None,
        title=f"{sc_sub_dict[visnum]}",
        spot_size=100,
        show=False,
        # vmin=vmin,
        # vmax=vmax,
        ax=ax,
    )


# %%
def plot_roc(visnum, adata, pred_sp, name, ax=None):
    """Plot ROC for a given visnum"""

    def layer_to_layer_number(x):
        for char in x:
            if char.isdigit():
                if int(char) in Ex_to_L_d[num_to_ex_d[visnum]]:
                    return 1
        return 0

    y_pred = pred_sp[:, visnum]
    y_true = adata.obs["spatialLIBD"].map(layer_to_layer_number).fillna(0)
    # print(y_true)
    # print(y_true.isna().sum())
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)

    return metrics.roc_auc_score(y_true, y_pred)


# %%
fig, ax = plt.subplots(
    nrows=1,
    ncols=len(st_sample_id_l),
    figsize=(3 * len(st_sample_id_l), 3),
    squeeze=False,
    constrained_layout=True,
    dpi=50,
)

cmap = mpl.cm.get_cmap("Accent_r")

color_range = list(
    np.linspace(
        0.125, 1, len(adata_spatialLIBD.obs.spatialLIBD.cat.categories), endpoint=True
    )
)
colors = [cmap(x) for x in color_range]

color_dict = defaultdict(lambda: "lightgrey")
for cat, color in zip(adata_spatialLIBD.obs.spatialLIBD.cat.categories, colors):
    color_dict[cat] = color

color_dict["NA"] = "lightgrey"

legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=color,
        markerfacecolor=color_dict[color],
        markersize=10,
    )
    for color in color_dict
]
fig.legend(bbox_to_anchor=(0, 0.5), handles=legend_elements, loc="center right")

for i, sample_id in enumerate(st_sample_id_l):
    sc.pl.spatial(
        adata_spatialLIBD_d[sample_id],
        img_key=None,
        color="spatialLIBD",
        palette=color_dict,
        size=1,
        title=sample_id,
        legend_loc=4,
        na_color="lightgrey",
        spot_size=100,
        show=False,
        ax=ax[0][i],
    )

    ax[0][i].axis("equal")
    ax[0][i].set_xlabel("")
    ax[0][i].set_ylabel("")


# fig.legend(loc=7)
fig.savefig(os.path.join(results_folder, "layers.png"), bbox_inches="tight", dpi=300)
plt.close()


# %%
realspots_d = {"da": {}}
if PRETRAINING:
    realspots_d["noda"] = {}

for sample_id in st_sample_id_l:
    fig, ax = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, dpi=10)

    for i, num in enumerate(numlist):
        plot_cellfraction(
            num, adata_spatialLIBD_d[sample_id], pred_sp_d[sample_id], ax.flat[i]
        )
        ax.flat[i].axis("equal")
        ax.flat[i].set_xlabel("")
        ax.flat[i].set_ylabel("")
    fig.suptitle(sample_id)

    fig.savefig(
        os.path.join(results_folder, f"{sample_id}_cellfraction.png"),
        bbox_inches="tight",
        dpi=300,
    )
    # fig.show()
    plt.close()

    fig, ax = plt.subplots(
        2, 5, figsize=(20, 8), constrained_layout=True, sharex=True, sharey=True, dpi=10
    )

    da_aucs = []
    if PRETRAINING:
        noda_aucs = []
    for i, num in enumerate(numlist):
        da_aucs.append(
            plot_roc(
                num,
                adata_spatialLIBD_d[sample_id],
                pred_sp_d[sample_id],
                MODEL_NAME,
                ax.flat[i],
            )
        )
        if PRETRAINING:
            noda_aucs.append(
                plot_roc(
                    num,
                    adata_spatialLIBD_d[sample_id],
                    pred_sp_noda_d[sample_id],
                    f"{MODEL_NAME}_wo_da",
                    ax.flat[i],
                )
            )

        ax.flat[i].plot(
            [0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k"
        )
        ax.flat[i].set_aspect("equal")
        ax.flat[i].set_xlim([0, 1])
        ax.flat[i].set_ylim([0, 1])

        ax.flat[i].set_title(f"{sc_sub_dict[num]}")

        if i >= len(numlist) - 5:
            ax.flat[i].set_xlabel("FPR")
        else:
            ax.flat[i].set_xlabel("")
        if i % 5 == 0:
            ax.flat[i].set_ylabel("TPR")
        else:
            ax.flat[i].set_ylabel("")

    realspots_d["da"][sample_id] = np.mean(da_aucs)
    if PRETRAINING:
        realspots_d["noda"][sample_id] = np.mean(noda_aucs)

    fig.suptitle(sample_id)
    fig.savefig(
        os.path.join(results_folder, f"{sample_id}_roc.png"),
        bbox_inches="tight",
        dpi=300,
    )
    # fig.show()
    plt.close()


# %%
def jsd(y_true, y_pred):

    kl = keras.losses.KLDivergence()
    m = 0.5 * (y_true + y_pred)
    return 0.5 * (kl(y_true, m) + kl(y_pred, m)).numpy()


# def jsd(y_true, y_pred):
#     return keras.losses.KLDivergence()(y_true, y_pred).numpy()
# %%
jsd_d = {"da": {}}
if PRETRAINING:
    jsd_d["noda"] = {}

for k in jsd_d:
    jsd_d[k] = {"train": {}, "val": {}, "test": {}}

for sample_id in st_sample_id_l:
    if TRAIN_USING_ALL_ST_SAMPLES:
        model = keras.models.load_model(
            os.path.join(advtrain_folder, "all_st_samps", "final_model")
        )
        if PRETRAINING:
            model_noda = keras.models.load_model(
                os.path.join(pretrain_folder, "all_st_samps", "final_model")
            )

    else:
        model = keras.models.load_model(
            os.path.join(advtrain_folder, sample_id, "final_model")
        )
        if PRETRAINING:
            model_noda = keras.models.load_model(
                os.path.join(pretrain_folder, sample_id, "final_model")
            )

    pred_mix_train = model.predict(sc_mix_d["train"])
    pred_mix_val = model.predict(sc_mix_d["val"])
    pred_mix_test = model.predict(sc_mix_d["test"])

    target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

    jsd_d["da"]["train"][sample_id] = jsd(lab_mix_d["train"], pred_mix_train)
    jsd_d["da"]["val"][sample_id] = jsd(lab_mix_d["val"], pred_mix_val)
    jsd_d["da"]["test"][sample_id] = jsd(lab_mix_d["test"], pred_mix_test)

    if PRETRAINING:
        pred_mix_train = model_noda.predict(sc_mix_d["train"])
        pred_mix_val = model_noda.predict(sc_mix_d["val"])
        pred_mix_test = model_noda.predict(sc_mix_d["test"])

        target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

        jsd_d["noda"]["train"][sample_id] = jsd(lab_mix_d["train"], pred_mix_train)
        jsd_d["noda"]["val"][sample_id] = jsd(lab_mix_d["val"], pred_mix_val)
        jsd_d["noda"]["test"][sample_id] = jsd(lab_mix_d["test"], pred_mix_test)


# %%
df_keys = [
    "Pseudospots (JS Divergence)",
    "RF50",
    "Real Spots (Mean AUC Ex1-10)",
]

if MILISI:
    df_keys.insert(2, "miLISI")


def gen_l_dfs(da):
    yield pd.DataFrame.from_dict(jsd_d[da], orient="columns")
    yield pd.DataFrame.from_dict(rf50_d[da], orient="columns")
    if MILISI:
        yield pd.DataFrame.from_dict(miLISI_d[da], orient="columns")
    yield pd.Series(realspots_d[da])
    return


da_dict_keys = ["da"]
da_df_keys = ["After DA"]
if PRETRAINING:
    da_dict_keys.insert(0, "noda")
    da_df_keys.insert(0, "Before DA")

results_df = pd.concat(
    [
        pd.concat(
            list(gen_l_dfs(da)),
            axis=1,
            keys=df_keys,
        )
        for da in da_dict_keys
    ],
    axis=0,
    keys=da_df_keys,
)

results_df.to_csv(os.path.join(results_folder, "results.csv"))
print(results_df)


# %%
