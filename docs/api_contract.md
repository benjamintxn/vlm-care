1. detector → produces:
   {
     "frame_id": int,
     "pedestrians": [
       {"bbox": [x1,y1,x2,y2], "confidence": float},
       …
     ]
   }

2. risk_engine → consumes detector output, produces:
   {
     "frame_id": int,
     "risk_score": float   # 0.0 (low) to 1.0 (high)
   }

3. explainer → consumes risk + detections, produces:
   {
     "frame_id": int,
     "explanation": str
   }
