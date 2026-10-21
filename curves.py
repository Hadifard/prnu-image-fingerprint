#-----------------------------------
#  IMAGE FINGERPRINT PRNU PROJECT 
#
# Author:
# hadi Sarhangi Fard
#
#-----------------------------------

"""Re-computes the scale-scan curves with a large patch and adds an untouched-crop tamper map (adds keys to assets_v2.pkl)."""
import pickle, numpy as np, cv2, os
import prnu_v2 as P
HERE="."; O="originals/"
A=pickle.load(open("assets_v2.pkl","rb"))
M=["20230504_131116","20230505_125151","20230527_214315","20240509_210037","20240518_203321","20240520_131048"]
G,N={},{}
G={};N={}
for m in M:
    g=P.load_gray(O+m+".jpg"); g=np.ascontiguousarray(np.rot90(g)) if g.shape[0]>g.shape[1] else g
    G[m]=g; N[m]=P.extract_noise(g)
def ref_without(m): 
    ks=[x for x in M if x!=m]; return P.Reference(P.estimate_fingerprint([G[x] for x in ks],[N[x] for x in ks]))
def curve(g, ref, rots, lo, hi, step, patch=1800):
    out=[]
    for s in np.arange(lo,hi+1e-9,step):
        out.append((float(s),max(float(P.match(g,ref,float(s),r,patch)["pce"]) for r in rots)))
    return out
lo,hi,st=0.628,0.660,0.001
A["curve_T2"]=dict(pts=curve(P.load_gray("tamper_set/T2_resize.jpg"),ref_without("20230505_125151"),(0,),lo,hi,st),true=0.644,rot=0)
print("T2 peak",max(A["curve_T2"]["pts"],key=lambda q:q[1]),flush=True)
A["curve_girl"]=dict(pts=curve(P.load_gray(O+"girl.jpg"),P.Reference(P.estimate_fingerprint(list(G.values()),list(N.values()))),(0,2),lo,hi,st),rot=0)
print("girl best",max(A["curve_girl"]["pts"],key=lambda q:q[1]),flush=True)
# untouched plain crop map
r=next(x for x in A["res"] if x["file"]=="T1_crop.jpg"); b=r["by_cam"]["A33"]
ref=ref_without("20240518_203321"); g=P.load_gray("tamper_set/T1_crop.jpg")
hyp=dict(rot=b["rot"],scale=b["scale"],shift=tuple(b["shift"]))
grid,rmap,gg,org=P.local_map(g,ref,hyp,win=128,stride=32)
sm=lambda a: cv2.resize(a,(a.shape[1]//4,a.shape[0]//4),interpolation=cv2.INTER_AREA).astype(np.float32)
A["maps"]["T1_crop.jpg"]=dict(rmap=sm(rmap),thumb=sm(gg),region=None,shape=gg.shape,med=float(np.median(rmap)),org=[int(org[0]),int(org[1])],div=4)
pickle.dump(A,open("assets_v2.pkl","wb"),protocol=4); print("saved",flush=True)
