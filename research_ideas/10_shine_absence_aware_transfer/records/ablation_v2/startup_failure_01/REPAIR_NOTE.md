Criterion diagnostic scalars may be integers. Logging accepts tensor and scalar values. Both initial attempts failed before backward/optimizer.step; no training design or loss changes.
Initial source preserved by freeze hash; precise replacement: float(v.detach()) -> float(v.detach()) if torch.is_tensor(v) else float(v), in original criterion logging only.
