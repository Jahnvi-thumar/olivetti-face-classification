python "C:\Users\JAHANVI\Downloads\plot_q3.py"#!/usr/bin/env python3
"""Reproduce Q3 bias/variance and depth-comparison figures/tables.

Run from repository root: python scripts/plot_q3.py
Uses the exact saved olivetti_split.npz; it never creates a new split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
FIGURES, RESULTS = ROOT / "figures", ROOT / "results"
SPLIT_PATH = ROOT / "olivetti_split.npz"
FIGURES.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
R, B_FINAL, B_DEPTH, MIN_LEAF = 30, 50, 100, 4
TREE_DEPTH, BOOST_DEPTH, BOOST_ROUNDS = None, 3, 100
DEPTH_SETTINGS = {"Shallow (depth=2)": 2, "Selected (unrestricted)": None, "Deep (depth=20)": 20}


def load_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} missing. Run Q1 setup first.")
    faces = fetch_olivetti_faces()
    X, y = faces.images.reshape(len(faces.images), -1), faces.target
    with np.load(SPLIT_PATH) as s:
        tr, te = s["train_indices"], s["test_indices"]
        seed = int(s["seed"]) if "seed" in s else 26016
    if len(tr) != 240 or len(te) != 80:
        raise ValueError("Unexpected split dimensions; expected 240 training and 80 test images.")
    return X[tr], y[tr], X[te], y[te], seed


def samme(X, y, depth, rounds, seed):
    classes = np.unique(y); K, n = len(classes), len(y)
    weights = np.ones(n) / n; learners, alphas = [], []
    for t in range(rounds):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed+t)
        tree.fit(X, y, sample_weight=weights)
        pred = tree.predict(X); error = float(weights[pred != y].sum())
        if error >= 1 - 1/K: break
        alpha = np.log((1-max(error, 1e-12))/max(error, 1e-12)) + np.log(K-1)
        learners.append(tree); alphas.append(alpha)
        if error > 0:
            weights[pred != y] *= np.exp(alpha); weights /= weights.sum()
        if error == 0: break
    return {"learners": learners, "alphas": np.asarray(alphas), "classes": classes}


def vectors_tree(model, X):
    out = np.zeros((len(X), 40))
    probs = model.predict_proba(X)
    for j, cls in enumerate(model.classes_): out[:, int(cls)] = probs[:, j]
    return out


def vectors_bag(trees, X):
    out = np.zeros((len(X), 40))
    for tree in trees:
        pred = tree.predict(X)
        out[np.arange(len(X)), pred.astype(int)] += 1
    return out / max(len(trees), 1)


def vectors_boost(model, X):
    out = np.zeros((len(X), 40)); total = model["alphas"].sum()
    for tree, alpha in zip(model["learners"], model["alphas"]):
        pred = tree.predict(X); out[np.arange(len(X)), pred.astype(int)] += alpha
    return out / total if total > 0 else out


def resample(rng, Xtr, ytr):
    selected = []
    for cls in np.unique(ytr):
        pos = np.where(ytr == cls)[0]
        selected.extend(rng.choice(pos, size=6, replace=True))
    selected = np.asarray(selected); rng.shuffle(selected)
    return Xtr[selected], ytr[selected]


def measure(predictions, accuracies, ytest):
    p = np.asarray(predictions)  # repetitions x N x 40
    mean_p = p.mean(axis=0)
    target = np.eye(40)[ytest]
    bias2 = np.mean(np.sum((mean_p-target)**2, axis=1))
    variance = np.mean(np.sum((p-mean_p[None,:,:])**2, axis=2))
    a = np.asarray(accuracies)
    return {"Bias squared": bias2, "Variance": variance,
            "Mean test accuracy": a.mean(), "SD test accuracy": a.std(ddof=0)}


def fit_bag(X, y, depth, min_leaf, B, rng, random_seed):
    trees = []
    for b in range(B):
        idx = rng.choice(len(X), size=len(X), replace=True)
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                   random_state=random_seed+b+1)
        t.fit(X[idx], y[idx]); trees.append(t)
    return trees


def plot_metric(df, xcol, metrics, filename, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(df[xcol].astype(str), df[col])
        ax.set_title(label); ax.set_xlabel(xcol); ax.set_ylabel(label); ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(title); fig.tight_layout()
    path = FIGURES / filename; fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", path)


def main():
    Xtr, ytr, Xte, yte, seed = load_data()
    classes = np.arange(40)
    # Q3(a): 30 resamples, same fixed clean test set.
    rng = np.random.default_rng(seed)
    names = ["Single Tree", "Bagging (50 trees)", "SAMME Boosting"]
    pred_hist = {n: [] for n in names}; acc_hist = {n: [] for n in names}
    for r in range(R):
        print(f"Q3(a): repetition {r+1}/{R}")
        Xr, yr = resample(rng, Xtr, ytr); rs = seed + 1000*(r+1)
        tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=MIN_LEAF, random_state=rs)
        tree.fit(Xr, yr)
        bag = fit_bag(Xr, yr, TREE_DEPTH, MIN_LEAF, B_FINAL, np.random.default_rng(rs), rs)
        boost = samme(Xr, yr, BOOST_DEPTH, BOOST_ROUNDS, rs)
        pv = [vectors_tree(tree, Xte), vectors_bag(bag, Xte), vectors_boost(boost, Xte)]
        for name, vec in zip(names, pv):
            pred_hist[name].append(vec)
            acc_hist[name].append(np.mean(np.argmax(vec, axis=1) == yte))
    rows = [{"Model": n, **measure(pred_hist[n], acc_hist[n], yte)} for n in names]
    df = pd.DataFrame(rows); df.to_csv(RESULTS/"q3-a-bias-variance.csv", index=False)
    print("\nQ3(a):\n", df.to_string(index=False))
    plot_metric(df, "Model", [("Bias squared","Bias squared"),("Variance","Variance")],
                "q3-a-bias-variance.png", "Q3(a): measured bias and variance")
    # Accuracy mean/std figure
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.bar(df["Model"], df["Mean test accuracy"], yerr=df["SD test accuracy"], capsize=4)
    ax.set_ylabel("Test accuracy"); ax.set_title("Q3(a): mean accuracy ± standard deviation")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    path=FIGURES/"q3-a-test-accuracy-stability.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); print("Saved",path)

    # Q3(b): shared resampled training set per repetition across all six models.
    rngb = np.random.default_rng(seed+30000)
    pred_b, acc_b = {}, {}
    for dname in DEPTH_SETTINGS:
        for kind in ["Single Tree", "Bagging 100"]:
            key=f"{kind} - {dname}"; pred_b[key]=[]; acc_b[key]=[]
    for r in range(R):
        print(f"Q3(b): repetition {r+1}/{R}")
        Xr, yr = resample(rngb, Xtr, ytr); rs=seed+40000+r*1000
        for dname, depth in DEPTH_SETTINGS.items():
            st=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=MIN_LEAF,random_state=rs)
            st.fit(Xr,yr); vec=vectors_tree(st,Xte); key=f"Single Tree - {dname}"
            pred_b[key].append(vec); acc_b[key].append(np.mean(vec.argmax(1)==yte))
            bag=fit_bag(Xr,yr,depth,MIN_LEAF,B_DEPTH,np.random.default_rng(rs+100),rs+100)
            vec=vectors_bag(bag,Xte); key=f"Bagging 100 - {dname}"
            pred_b[key].append(vec); acc_b[key].append(np.mean(vec.argmax(1)==yte))
    rb=[]
    for key in pred_b: rb.append({"Model":key,**measure(pred_b[key],acc_b[key],yte)})
    dfb=pd.DataFrame(rb); dfb.to_csv(RESULTS/"q3-b-depth-bias-variance.csv",index=False)
    print("\nQ3(b):\n",dfb.to_string(index=False))
    # paired single vs bagging depth summaries
    comp=[]
    for dname in DEPTH_SETTINGS:
        s=dfb[dfb.Model==f"Single Tree - {dname}"].iloc[0]
        b=dfb[dfb.Model==f"Bagging 100 - {dname}"].iloc[0]
        comp.append({"Depth":dname,"Single bias squared":s["Bias squared"],"Bagging bias squared":b["Bias squared"],
          "Single variance":s["Variance"],"Bagging variance":b["Variance"],
          "Variance removed":s["Variance"]-b["Variance"],
          "Variance reduction (%)":100*(s["Variance"]-b["Variance"])/s["Variance"] if s["Variance"] else np.nan,
          "Single accuracy":s["Mean test accuracy"],"Bagging accuracy":b["Mean test accuracy"],
          "Accuracy improvement":b["Mean test accuracy"]-s["Mean test accuracy"]})
    cdf=pd.DataFrame(comp); cdf.to_csv(RESULTS/"q3-b-variance-reduction.csv",index=False)
    print("\nQ3(b) variance reduction:\n",cdf.to_string(index=False))
    fig, axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax, col, title in zip(axes,["Variance removed","Accuracy improvement","Variance reduction (%)"],
                              ["Variance removed by bagging","Test accuracy improvement","Variance reduction (%)"]):
        ax.bar(cdf["Depth"],cdf[col]); ax.set_title(title); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); path=FIGURES/"q3-b-depth-bagging-comparison.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); print("Saved",path)

    # Q3(c): one fixed resampled set for leaf occupancy and tree disagreement, matching notebook logic.
    rs=seed+50000; Xr,yr=resample(np.random.default_rng(rs),Xtr,ytr); rowsc=[]
    for dname,depth in DEPTH_SETTINGS.items():
        st=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=MIN_LEAF,random_state=rs); st.fit(Xr,yr)
        leaf_ids=np.where(st.tree_.children_left==-1)[0]; leaf_counts=st.tree_.n_node_samples[leaf_ids]
        bag=fit_bag(Xr,yr,depth,MIN_LEAF,100,np.random.default_rng(rs+100),rs+100)
        occ=[]; singleton=[] 
        for t in bag:
            lids=np.where(t.tree_.children_left==-1)[0]; counts=t.tree_.n_node_samples[lids]
            occ.append(counts.mean()); singleton.append(np.mean(counts==1))
        pp=np.asarray([t.predict(Xte) for t in bag]); dis=[]
        total_pairs=100*99/2
        for i in range(len(Xte)):
            _, counts=np.unique(pp[:,i],return_counts=True)
            agreeing=np.sum(counts*(counts-1))/2
            dis.append(1-agreeing/total_pairs)
        sr=dfb[dfb.Model==f"Single Tree - {dname}"].iloc[0]
        br=dfb[dfb.Model==f"Bagging 100 - {dname}"].iloc[0]
        rowsc.append({"Depth":dname,"Single accuracy":sr["Mean test accuracy"],"Bagged accuracy":br["Mean test accuracy"],
          "Accuracy improvement":br["Mean test accuracy"]-sr["Mean test accuracy"],
          "Single bias squared":sr["Bias squared"],"Bagged bias squared":br["Bias squared"],
          "Single variance":sr["Variance"],"Bagged variance":br["Variance"],
          "Variance reduction":sr["Variance"]-br["Variance"],
          "Single mean leaf occupancy":leaf_counts.mean(),"Single singleton leaf fraction":np.mean(leaf_counts==1),
          "Bagged mean leaf occupancy":np.mean(occ),"Bagged singleton leaf fraction":np.mean(singleton),
          "Pairwise disagreement":np.mean(dis)})
    dfc=pd.DataFrame(rowsc); dfc.to_csv(RESULTS/"q3-c-depth-occupancy-disagreement.csv",index=False)
    print("\nQ3(c):\n",dfc.to_string(index=False))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,col,title in zip(axes,["Single singleton leaf fraction","Pairwise disagreement","Accuracy improvement"],
                            ["Single-tree singleton leaf fraction","Bagged-tree pairwise disagreement","Bagging accuracy improvement"]):
        ax.plot(dfc["Depth"],dfc[col],marker="o"); ax.set_title(title); ax.tick_params(axis="x",rotation=20); ax.grid(True,alpha=.3)
    fig.tight_layout(); path=FIGURES/"q3-c-depth-occupancy-disagreement.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); print("Saved",path)

    # Q3(d) final table uses fresh evaluation on clean/masked test data and measured Q3(a) bias/variance.
    Xm=Xte.reshape(-1,64,64).copy(); Xm[:,24:40,24:40]=0; Xm=Xm.reshape(len(Xm),-1)
    final_tree=DecisionTreeClassifier(max_depth=TREE_DEPTH,min_samples_leaf=MIN_LEAF,random_state=seed).fit(Xtr,ytr)
    final_bag=fit_bag(Xtr,ytr,TREE_DEPTH,MIN_LEAF,B_FINAL,np.random.default_rng(seed),seed)
    final_boost=samme(Xtr,ytr,BOOST_DEPTH,BOOST_ROUNDS,seed)
    model_objs=[("Single Tree",final_tree),("Bagging (50 trees)",final_bag),("SAMME Boosting",final_boost)]
    out=[]
    for name,obj in model_objs:
        if name=="Single Tree": clean=vectors_tree(obj,Xte); masked=vectors_tree(obj,Xm)
        elif name=="Bagging (50 trees)": clean=vectors_bag(obj,Xte); masked=vectors_bag(obj,Xm)
        else: clean=vectors_boost(obj,Xte); masked=vectors_boost(obj,Xm)
        row=df[df.Model==name].iloc[0]
        out.append({"Model":name,"Clean test accuracy":np.mean(clean.argmax(1)==yte),
                    "Masked test accuracy":np.mean(masked.argmax(1)==yte),
                    "Mean clean winning vote/probability":np.mean(clean.max(1)),
                    "Mean masked winning vote/probability":np.mean(masked.max(1)),
                    "Bias squared":row["Bias squared"],"Variance":row["Variance"]})
    final=pd.DataFrame(out); final.to_csv(RESULTS/"q3-d-final-comparison.csv",index=False)
    print("\nQ3(d) final comparison:\n",final.to_string(index=False))
    print("\nAll Q3 plots and tables saved under figures/ and results/.")

if __name__=="__main__":
    main()
