"""Label-free fusion from the agreement of two fixed views per checkpoint."""
from __future__ import annotations
import numpy as np


def disagreement(first,second):
    return np.linalg.norm(np.asarray(first,dtype=np.float64).reshape(-1,2)-
                          np.asarray(second,dtype=np.float64).reshape(-1,2),axis=-1)


def infer_floor(sources,id_key):
    maps=[{r[id_key]:r for r in source} for source in sources]
    if any(set(m)!=set(maps[0]) for m in maps):raise ValueError("view inventory mismatch")
    tasks={}
    for identifier,a in maps[0].items():
        values=[disagreement(maps[j][identifier]["predicted_points_pixels"],
                             maps[j+1][identifier]["predicted_points_pixels"]).mean() for j in [0,2]]
        tasks.setdefault(a["task_id"],[]).extend(values)
    if len(tasks)!=9:raise ValueError("all nine tasks required for feature-only scale calibration")
    return max(1e-3,float(np.median([np.median(values) for values in tasks.values()])))


def fuse(sources,kind,floor,id_key="manifest_row_id"):
    if len(sources)!=4 or floor<=0:raise ValueError("four fixed views and positive floor required")
    maps=[{r[id_key]:r for r in source} for source in sources]
    if any(set(m)!=set(maps[0]) or len(m)!=len(sources[0]) for m in maps):raise ValueError("view inventory mismatch")
    records=[]
    for row in sources[0]:
        peers=[m[row[id_key]] for m in maps]
        if any(p["task_id"]!=row["task_id"] for p in peers):raise ValueError("view task mismatch")
        points=np.stack([np.asarray(p["predicted_points_pixels"],dtype=np.float64).reshape(-1,2) for p in peers])
        if not np.isfinite(points).all():raise ValueError("nonfinite predictions")
        # Match the existing frozen single-model serialization before cross-model fusion.
        old=np.round(points[:2].mean(0),4);new=np.round(points[2:].mean(0),4)
        if kind=="new_single":output=new
        elif kind=="equal_mean":output=(old+new)/2
        elif kind in {"case_consistency","point_consistency"}:
            d0=np.linalg.norm(points[0]-points[1],axis=-1)
            d1=np.linalg.norm(points[2]-points[3],axis=-1)
            if kind=="case_consistency":d0,d1=d0.mean(),d1.mean()
            p0=1/np.sqrt(d0*d0+floor*floor);p1=1/np.sqrt(d1*d1+floor*floor)
            weight=np.clip(p0/(p0+p1),.25,.75)
            if np.ndim(weight):weight=weight[:,None]
            output=weight*old+(1-weight)*new
        elif kind=="geometric_median":
            output=points.mean(0)
            for _ in range(50):
                weights=1/np.maximum(np.linalg.norm(points-output[None],axis=-1),1e-6)
                output=(weights[...,None]*points).sum(0)/weights.sum(0)[:,None]
        else:raise ValueError("undeclared fusion kind")
        records.append({**row,"predicted_points_pixels":np.round(output,4).reshape(-1).tolist()})
    return records
