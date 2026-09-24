import csv, math, itertools
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
runs = {"S1":"dashmamba_stage1_track_a_test","S2":"dashmamba_stage2_track_a_test",
        "RVRT":"rvrt_track_a_test","FastDVD":"fastdvdnet_track_a_test",
        "BVSRpp":"bvrpp_track_a_test"}
data={}
for k,d in runs.items():
    f=R/d/"per_clip.csv"
    if not f.exists(): continue
    m={}
    for r in csv.DictReader(f.open()):
        m[(r["kind"],r["level"],r["clip_stem"])]={
            "psnr":float(r["psnr"]),"ssim":float(r["ssim"]),"tof":float(r["tof"])}
    data[k]=m
axes=[("gaussian","high"),("gaussian","medium"),("gaussian","low"),("poisson_gaussian","realistic")]

def paired(a,b,ax,metric="psnr"):
    A,B=data[a],data[b]
    keys=[k for k in A if k[0]==ax[0] and k[1]==ax[1] and k in B]
    d=[A[k][metric]-B[k][metric] for k in keys]
    n=len(d)
    if n<2: return None
    mu=sum(d)/n
    sd=math.sqrt(sum((x-mu)**2 for x in d)/(n-1))
    se=sd/math.sqrt(n)
    t=mu/se if se else float('nan')
    ci=1.994*se   # t_.975, df=70
    wins=sum(1 for x in d if x>0)
    return n,mu,mu-ci,mu+ci,t,wins

print("=== Stage-1 (pretrain-only, FAIR vs baselines) : paired per-clip PSNR ===")
for ax in axes:
    print(f"\n-- {ax[0]}/{ax[1]}")
    for b in ("RVRT","FastDVD","S2"):
        r=paired("S1",b,ax)
        if r:
            n,mu,lo,hi,t,w=r
            sig = "SIG" if (lo>0 or hi<0) else "ns "
            print(f"   S1 vs {b:8s} n={n:3d}  {mu:+6.2f} dB  CI[{lo:+6.2f},{hi:+6.2f}]  t={t:+6.2f}  wins {w}/{n}  {sig}")

print("\n=== tOF (lower better; negative = S1 better) ===")
for ax in axes:
    row=f"  {ax[0]}/{ax[1]:10s}"
    for b in ("RVRT","FastDVD"):
        r=paired("S1",b,ax,"tof")
        if r: row+=f"  vs {b}: {r[1]:+5.2f} (t={r[4]:+5.1f})"
    print(row)

print("\n=== SSIM ===")
for ax in axes:
    row=f"  {ax[0]}/{ax[1]:10s}"
    for b in ("RVRT","FastDVD"):
        r=paired("S1",b,ax,"ssim")
        if r: row+=f"  vs {b}: {r[1]:+.4f} (t={r[4]:+5.1f})"
    print(row)
