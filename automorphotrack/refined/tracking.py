"""One-to-one, motion-assisted particle association with explicit ambiguity."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class TrackingConfig:
    max_distance_px: float = 6.0
    ambiguity_margin_px: float = 0.5
    prediction_weight: float = 0.0
    max_gap_frames: int = 0

    def __post_init__(self):
        if not math.isfinite(self.max_distance_px) or self.max_distance_px <= 0:
            raise ValueError("max_distance_px must be positive and finite")
        if not math.isfinite(self.ambiguity_margin_px) or self.ambiguity_margin_px < 0:
            raise ValueError("ambiguity_margin_px must be nonnegative and finite")
        if not math.isfinite(self.prediction_weight) or not 0 <= self.prediction_weight <= 1:
            raise ValueError("prediction_weight must lie between zero and one")
        if isinstance(self.max_gap_frames, bool) or not isinstance(self.max_gap_frames, int) or not 0 <= self.max_gap_frames <= 2:
            raise ValueError("max_gap_frames must be an integer from zero through two")


COLUMNS = ["Frame", "Track_ID", "Label", "X", "Y", "Link_Status", "Link_Distance_px",
           "Previous_Candidate_Count", "Current_Candidate_Count", "Delta_Frame",
           "Prediction_Error_px", "Assignment_Margin_px"]


def link_spots(centroids, config=None, labels_per_frame=None):
    """Keep gaps and ambiguous links visible; never interpolate missing positions.

    Accepted adjacent links require a clear distance margin in both directions.
    Gap links have a separate status and are excluded from speed estimation.
    """
    config = config or TrackingConfig()
    active, next_id, rows = {}, 0, []
    for frame, points in enumerate(centroids):
        current = np.asarray(points, dtype=float)
        if current.size == 0:
            current = np.empty((0, 2))
        if current.ndim != 2 or current.shape[1] != 2 or not np.isfinite(current).all():
            raise ValueError("Each frame must contain finite N by 2 row/column positions")
        if labels_per_frame is not None and len(labels_per_frame[frame]) != len(current):
            raise ValueError("Labels and positions must have equal lengths")
        active = {k:v for k,v in active.items() if frame-v['frame'] <= config.max_gap_frames+1}
        ids = sorted(active)
        links = {}
        allowed = np.zeros((len(ids),len(current)),bool)
        distances = np.empty(allowed.shape)
        raw = np.empty(allowed.shape)
        if ids and len(current):
            old = np.array([active[k]['point'] for k in ids])
            dt = np.array([frame-active[k]['frame'] for k in ids])
            predicted = old + np.array([active[k]['velocity'] for k in ids]) * dt[:,None] * config.prediction_weight
            distances = cdist(predicted,current)
            raw = cdist(old,current)
            gates = config.max_distance_px * dt
            allowed = (raw <= gates[:,None]) & (distances <= gates[:,None])
            n,m=allowed.shape
            # Dummy assignments permit births and losses instead of forcing links.
            costs=np.full((n+m,m+n),1e12)
            costs[:n,:m]=np.where(allowed,(distances/gates[:,None])**2,1e12)
            costs[:n,m:]=0.5
            costs[n:,:m]=0.5
            costs[n:,m:]=0
            a,b=linear_sum_assignment(costs)
            links={int(j):int(i) for i,j in zip(a,b) if i<n and j<m and allowed[i,j]}
        for j, point in enumerate(current):
            if j in links:
                i=links[j];tid=ids[i];previous=active[tid]
                dt=frame-previous['frame']
                other_row=np.where(allowed[i],distances[i],np.inf).copy();other_row[j]=np.inf
                other_col=np.where(allowed[:,j],distances[:,j],np.inf).copy();other_col[i]=np.inf
                margin=min(other_row.min(initial=np.inf),other_col.min(initial=np.inf))-distances[i,j]
                clear=margin > config.ambiguity_margin_px
                status=('matched' if dt==1 else 'gap') if clear else 'ambiguous'
                velocity=(point-previous['point'])/dt if clear else np.zeros(2)
                distance=float(raw[i,j]);error=float(distances[i,j])
                pc=int(allowed[i].sum());cc=int(allowed[:,j].sum())
            else:
                tid=next_id;next_id+=1;dt=0
                status='start' if frame==0 else 'birth_or_unmatched'
                velocity=np.zeros(2);distance=error=margin=np.nan;pc=0
                cc=int(allowed[:,j].sum())
            active[tid]=dict(frame=frame,point=point.copy(),velocity=velocity)
            label=labels_per_frame[frame][j] if labels_per_frame is not None else j+1
            rows.append([frame,tid,label,float(point[1]),float(point[0]),status,distance,pc,cc,dt,error,margin])
    return pd.DataFrame(rows,columns=COLUMNS)
