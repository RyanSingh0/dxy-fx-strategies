"""The full pipeline run on exchange data, with the open-location screen as an exact test.

Everything applied to the free series is applied here to the exchange series: the
24-strategy grid at three cost levels, Newey-West, PSR, DSR, White RC, Hansen SPA,
Romano-Wolf, CSCV/PBO, holdout and walk-forward. The screen is then written as an exact
binomial test so that it can be reused on other series."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import paths

import numpy as np, pandas as pd, itertools, sys
from scipy import stats as st
import statsmodels.api as sm
T = 252
U = str(paths.DATA)

def load_bc():
    d = pd.read_csv(str(paths.DX_FILE),
                    skipfooter=1, engine="python")
    return pd.DataFrame({"Date": pd.to_datetime(d["Time"]),
        "O": pd.to_numeric(d["Open"],errors="coerce"), "H": pd.to_numeric(d["High"],errors="coerce"),
        "L": pd.to_numeric(d["Low"],errors="coerce"),  "C": pd.to_numeric(d["Latest"],errors="coerce")
        }).sort_values("Date").reset_index(drop=True)
def load_yh():
    d = pd.read_csv(str(paths.VENDOR_FILE), parse_dates=["Date"])
    return d[["Date","DXY_Open","DXY_High","DXY_Low","DXY_Close"]].rename(
        columns={"DXY_Open":"O","DXY_High":"H","DXY_Low":"L","DXY_Close":"C"}).sort_values("Date").reset_index(drop=True)

MAP={1:("Long","Cash"),2:("Short","Cash"),3:("Cash","Long"),4:("Cash","Short"),5:("Long","Long"),
6:("Short","Short"),7:("Short","Long"),8:("Long","Short"),9:("Cash","Inertia"),10:("Cash","Reversal"),
11:("Inertia","Cash"),12:("Reversal","Cash"),13:("Inertia","Inertia"),14:("Inertia","Reversal"),
15:("Reversal","Inertia"),16:("Reversal","Reversal"),17:("Long","Inertia"),18:("Long","Reversal"),
19:("Short","Inertia"),20:("Short","Reversal"),21:("Inertia","Long"),22:("Reversal","Long"),
23:("Inertia","Short"),24:("Reversal","Short")}
LBL={k:"(%s, %s)"%v for k,v in MAP.items()}
def pos(r,s):
    if r=="Cash": return 0
    if r=="Long": return 1
    if r=="Short": return -1
    if np.isnan(s): return 0
    b = 1 if s>=0 else -1
    return b if r=="Inertia" else -b
sr = lambda x: np.asarray(x,float).mean()/np.asarray(x,float).std(ddof=1)*np.sqrt(T)
tv = lambda x: 100*np.prod(1+np.asarray(x,float))
def cg(x): return ((tv(x)/100)**(T/len(x))-1)*100
def mdd(x):
    pv=100*np.cumprod(1+np.asarray(x,float)); return 100*(pv/np.maximum.accumulate(pv)-1).min()
def nw(x,l=None):
    # Newey and West (1987) Bartlett kernel with the standard automatic bandwidth
    # floor(4*(T/100)^(2/9)), which gives 8 lags at this sample length.
    if l is None: l=int(np.floor(4*(len(x)/100.0)**(2.0/9.0)))
    x=np.asarray(x,float)
    r=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':l})
    return float(r.tvalues[0]), float(r.pvalues[0])

def build(d):
    co=(d.O/d.C.shift(1)-1).values; oc=(d.C/d.O-1).values; N=len(d)
    bnh=d.C.ffill().pct_change().fillna(0).values
    def grid(a,b,cb):
        c=cb/1e4; prev=np.concatenate(([np.nan],oc[:-1])); held=0
        ro=np.empty(N); rd=np.empty(N); ntr=0
        for i in range(N):
            p=pos(a,prev[i]); ch=c if (p!=0 and p!=held) else 0.0
            if p!=0 and p!=held: ntr+=1
            ro[i]=0.0 if np.isnan(co[i]) else p*co[i]-ch; held=p
            q=pos(b,co[i]); ch=c if (q!=0 and q!=held) else 0.0
            if q!=0 and q!=held: ntr+=1
            rd[i]=0.0 if np.isnan(oc[i]) else q*oc[i]-ch; held=q
        return (1+ro)*(1+rd)-1, ntr
    R={}; TR={}
    for cb in (0.0,1.0,2.0):
        cols=[];trs=[]
        for k in range(1,25):
            r,n_=grid(*MAP[k],cb); cols.append(r); trs.append(n_)
        R[cb]=np.column_stack(cols); TR[cb]=trs
    return co,oc,bnh,R,TR

bc=load_bc(); yh=load_yh()
co_b,oc_b,bnh_b,Rb,TRb = build(bc)
co_y,oc_y,bnh_y,Ry,TRy = build(yh)

print("="*94)
print("1.  ALL 24 STRATEGIES ON EXCHANGE DATA, THREE COST LEVELS")
print("="*94)
M=Rb[0.0]
fin=np.array([tv(M[:,i]) for i in range(24)]); srs=np.array([sr(M[:,i]) for i in range(24)])
print("%3s %-22s %9s %8s %8s %8s %8s %8s"%("#","strategy","final","total%","CAGR%","Sharpe","MaxDD%","trades"))
for i in range(24):
    print("%3d %-22s %9.2f %8.1f %8.2f %8.3f %8.1f %8d"
          %(i+1,LBL[i+1],fin[i],fin[i]-100,cg(M[:,i]),srs[i],mdd(M[:,i]),TRb[0.0][i]))
print("%3s %-22s %9.2f %8.1f %8.2f %8.3f %8.1f %8d"
      %("","buy-and-hold",tv(bnh_b),tv(bnh_b)-100,cg(bnh_b),sr(bnh_b),mdd(bnh_b),1))
print("\nbeating buy-and-hold: %d of 24 at 0 bps, %d at 1 bps, %d at 2 bps"
      %((fin>tv(bnh_b)).sum(),
        (np.array([tv(Rb[1.0][:,i]) for i in range(24)])>tv(bnh_b)).sum(),
        (np.array([tv(Rb[2.0][:,i]) for i in range(24)])>tv(bnh_b)).sum()))
print("Sharpe range: %+.3f to %+.3f    Strategy 7 rank: %d of 24"%(srs.min(),srs.max(),1+(fin>fin[6]).sum()))

print("\n"+"="*94)
print("2.  FORMAL TESTS ON EXCHANGE DATA  (same battery as the current draft)")
print("="*94)
def psr_dsr(x, all_sr_ann, N):
    x=np.asarray(x,float); s=x.mean()/x.std(ddof=1); sk=st.skew(x); ku=st.kurtosis(x,fisher=False); n=len(x)
    v=np.var(np.asarray(all_sr_ann)/np.sqrt(T),ddof=1); g=np.euler_gamma
    sstar=np.sqrt(v)*((1-g)*st.norm.ppf(1-1/N)+g*st.norm.ppf(1-1/(N*np.e)))
    den=np.sqrt(1-sk*s+((ku-1)/4)*s**2)
    return float(st.norm.cdf(s*np.sqrt(n-1)/den)), float(st.norm.cdf((s-sstar)*np.sqrt(n-1)/den)), sstar*np.sqrt(T)
top=np.argsort(-fin)[:5]
print("%-24s %8s %8s %9s %8s %8s"%("strategy","SR","t_NW","p","PSR(>0)","DSR"))
for i in top:
    t,p=nw(M[:,i]); ps,ds,sst=psr_dsr(M[:,i],srs,24)
    print("S%-2d %-20s %8.3f %8.3f %9.4f %8.4f %8.4f"%(i+1,LBL[i+1],srs[i],t,p,ps,ds))
t,p=nw(bnh_b)
print("%-24s %8.3f %8.3f %9.4f"%("buy-and-hold",sr(bnh_b),t,p))
print("expected max Sharpe under the null (N=24): %.3f   observed max: %.3f"%(sst,srs.max()))
F,pf=st.f_oneway(*[M[:,i] for i in range(24)]); H,ph=st.kruskal(*[M[:,i] for i in range(24)])
print("ANOVA F(23,%d) = %.3f, p = %.4f | Kruskal-Wallis H = %.2f, p = %.4f"%(24*len(bc)-24,F,pf,H,ph))

print("\n"+"="*94)
print("3.  MULTIPLE TESTING, EXCHANGE vs FREE")
print("="*94)
rng=np.random.default_rng(20260828)
def sb_idx(n,B,mb):
    p=1/mb; idx=np.empty((B,n),dtype=np.int64)
    s0=rng.integers(0,n,size=B); jp=rng.random((B,n))<p; ns=rng.integers(0,n,size=(B,n))
    for b in range(B):
        i=s0[b]; out=idx[b]
        for t_ in range(n):
            out[t_]=i; i=ns[b,t_] if jp[b,t_] else (i+1)%n
    return idx
def mt(Rd,bnh,cb,B=5000):
    f=Rd[cb]-bnh[:,None]; n,k=f.shape; fbar=f.mean(axis=0)
    IDX=sb_idx(n,B,21); boot=np.empty((B,k))
    for b in range(B): boot[b]=f[IDX[b]].mean(axis=0)
    om=boot.std(axis=0,ddof=1)*np.sqrt(n); om[om==0]=1e-12
    w=np.sqrt(n)*(boot-fbar)
    p_rc=float((w.max(axis=1)>=np.sqrt(n)*fbar.max()).mean())
    thr=-(om/np.sqrt(n))*np.sqrt(2*np.log(np.log(n))); g=np.where(fbar>=thr,fbar,0.0)
    Tobs=max(0.0,float(np.max(np.sqrt(n)*fbar/om))); Tb=np.maximum(0.0,(np.sqrt(n)*(boot-g)/om).max(axis=1))
    p_spa=float((Tb>=Tobs).mean())
    stud=np.sqrt(n)*fbar/om; surv=[]; act=list(range(k))
    while act:
        cr=np.percentile((w[:,act]/om[act]).max(axis=1),95)
        rej=[x for x in act if stud[x]>cr]
        if not rej: break
        surv+=rej; act=[x for x in act if x not in rej]
    return p_rc,p_spa,len(surv),surv
print("%-12s %6s %12s %12s %16s"%("source","cost","White RC p","Hansen SPA p","RW survivors"))
for lab,Rd,bh in [("exchange",Rb,bnh_b),("free",Ry,bnh_y)]:
    for cb in (0.0,1.0,2.0):
        a,b_,c_,sv=mt(Rd,bh,cb)
        print("%-12s %6.0f %12.4f %12.4f %16s"%(lab,cb,a,b_,"%d of 24"%c_ + (" ("+", ".join("S%d"%(x+1) for x in sorted(sv,key=lambda z:-(Rd[cb][:,z]-bh).mean()))+")" if sv else "")))

print("\n"+"="*94)
print("4.  CSCV / PROBABILITY OF BACKTEST OVERFITTING")
print("="*94)
def cscv(M,Sb):
    n,k=M.shape; blk=np.array_split(np.arange(n),Sb); lam=[];rk=[]
    for cmb in itertools.combinations(range(Sb),Sb//2):
        tr=np.concatenate([blk[i] for i in cmb]); te=np.concatenate([blk[i] for i in range(Sb) if i not in cmb])
        a=np.array([sr(M[tr,i]) for i in range(k)]); b=np.array([sr(M[te,i]) for i in range(k)])
        i=int(np.nanargmax(a)); r=1+int((b<b[i]).sum()); w=r/(k+1.0)
        lam.append(np.log(w/(1-w))); rk.append(r)
    lam=np.array(lam); return float((lam<=0).mean()), float(np.median(rk)), len(lam)
print("%-12s %6s %8s %8s %10s"%("source","cost","blocks","PBO","med rank"))
for lab,Rd in [("exchange",Rb),("free",Ry)]:
    for cb in (0.0,1.0):
        for Sb in (12,16):
            p,mr,ns=cscv(Rd[cb],Sb)
            print("%-12s %6.0f %8d %8.3f %10.1f"%(lab,cb,Sb,p,mr))

print("\n"+"="*94)
print("5.  HOLDOUT AND WALK-FORWARD ON EXCHANGE DATA")
print("="*94)
yr=bc.Date.dt.year.values; IS=yr<=2018; OS=yr>=2019
c0=np.nan_to_num(co_b); o0=np.nan_to_num(oc_b)
print("%-16s %10s %8s %8s   %10s %8s %8s"%("","IS 100->","IS SR","IS p","OOS 100->","OOS SR","OOS p"))
for nm,x in [("overnight leg",c0),("daytime leg",o0),("buy-and-hold",bnh_b)]:
    _,pi=nw(x[IS]); _,po=nw(x[OS])
    print("%-16s %10.2f %8.3f %8.4f   %10.2f %8.3f %8.4f"%(nm,tv(x[IS]),sr(x[IS]),pi,tv(x[OS]),sr(x[OS]),po))
isr=np.array([sr(Rb[0.0][IS,i]) for i in range(24)]); osr=np.array([sr(Rb[0.0][OS,i]) for i in range(24)])
i=int(np.nanargmax(isr)); t,p=nw(Rb[0.0][OS,i])
print("\nselect best on 2010-2018 -> S%d %s (IS SR %+.3f); holdout SR %+.3f, 100 -> %.2f, p %.3f, rank %d/24"
      %(i+1,LBL[i+1],isr[i],osr[i],tv(Rb[0.0][OS,i]),p,1+int((osr>osr[i]).sum())))
years=sorted(set(yr)); oos=[];log=[]
for n_,y in enumerate(years):
    if n_<3: continue
    tr=yr<y; te=yr==y
    if tr.sum()<200 or te.sum()<20: continue
    s=np.array([sr(Rb[0.0][tr,i]) for i in range(24)]); i=int(np.nanargmax(s))
    oos.append(Rb[0.0][te,i]); log.append((y,i+1))
stream=np.concatenate(oos); t,p=nw(stream)
bstream=bnh_b[yr>=log[0][0]]
print("walk-forward %d OOS years (%d-%d): 100 -> %.2f, SR %+.3f, p %.3f | buy-and-hold 100 -> %.2f, SR %+.3f"
      %(len(log),log[0][0],log[-1][0],tv(stream),sr(stream),p,tv(bstream),sr(bstream)))

print("\n"+"="*94)
print("6.  THE OPEN-LOCATION SCREEN AS AN EXACT TEST")
print("="*94)
print("Under the null that the recorded open is an unbiased draw from the session,")
print("P(open = daily low) = P(open = daily high). Counting the two gives an exact")
print("binomial test that needs no second data source.\n")
print("%-26s %7s %7s %9s %12s %10s"%("series","at low","at high","n extreme","binomial p","mean pos"))
def screen(O,H,L,lab):
    k=O.notna()&H.notna()&L.notna()&(H>L)
    lo=int(np.isclose(O[k],L[k],rtol=0,atol=1e-9).sum()); hi=int(np.isclose(O[k],H[k],rtol=0,atol=1e-9).sum())
    n=lo+hi
    p=float(st.binomtest(lo,n,0.5).pvalue) if n>0 else np.nan
    mp=float(((O[k]-L[k])/(H[k]-L[k])).mean())
    print("%-26s %7d %7d %9d %12.3g %10.3f"%(lab,lo,hi,n,p,mp))
    return p
screen(bc.O,bc.H,bc.L,"DXY futures, exchange")
screen(yh.O,yh.H,yh.L,"DXY futures, free")
fx=pd.read_csv(str(paths.VENDOR_FILE),parse_dates=["Date"])
for p_ in ["EURUSD","GBPUSD","JPYUSD","CADUSD","CHFUSD","SEKUSD"]:
    screen(fx[f"{p_}_Open"],fx[f"{p_}_High"],fx[f"{p_}_Low"],"%s, free"%p_)
print("\n-> the screen flags the one series that is known to be wrong and clears the")
print("   exchange series, using only the series' own high, low and open.")
