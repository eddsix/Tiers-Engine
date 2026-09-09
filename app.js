const TIERS=[27,13,36,11,30,8,23,10,5,24,16,33];
const WHEEL=[0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];
const KEY="tiers-engine-history-v1";
let history=JSON.parse(localStorage.getItem(KEY)||"[]");

const $=id=>document.getElementById(id);
function isTiers(n){return TIERS.includes(n)}
function save(){localStorage.setItem(KEY,JSON.stringify(history))}
function pct(x){return `${(x*100).toFixed(1)}%`}
function rates(arr){
  if(!arr.length)return null;
  return arr.filter(isTiers).length/arr.length
}
function streak(arr, wanted){
  let n=0; for(const x of arr){if(isTiers(x)===wanted)n++;else break} return n
}
function intervals(arr){
  const pos=[]; arr.forEach((x,i)=>{if(isTiers(x))pos.push(i)});
  if(pos.length<2)return [];
  const out=[]; for(let i=0;i<pos.length-1;i++)out.push(pos[i]-pos[i+1]*-1-1);
  return out;
}
function wheelDistance(a,b){
  const ia=WHEEL.indexOf(a),ib=WHEEL.indexOf(b),d=Math.abs(ia-ib);
  return Math.min(d,WHEEL.length-d)
}
function direction(a,b){
  const ia=WHEEL.indexOf(a),ib=WHEEL.indexOf(b),n=WHEEL.length;
  const cw=(ib-ia+n)%n, ccw=(ia-ib+n)%n;
  if(cw===ccw)return "Neutral";
  return cw<ccw?"Clockwise":"Counterclockwise";
}
function model(arr){
  if(arr.length<12)return {tiers:.5,confidence:"—",ready:false,note:"At least 12 results are required before estimating a signal."};
  const windows=[10,20,50,100,250].map(w=>arr.slice(0,w)).filter(x=>x.length>=5);
  const base=rates(arr)||.5;
  const wr=windows.map(rates);
  // Reliability-aware blend: larger samples receive more weight.
  let p=base, weight=1;
  windows.forEach((w,i)=>{const rel=Math.min(1,w.length/50);p+=wr[i]*rel;weight+=rel});
  p/=weight;
  // Transition context: empirical next-state rate after the latest state.
  if(arr.length>=20){
    const last=isTiers(arr[0]); let count=0,hit=0;
    for(let i=1;i<arr.length;i++) if(isTiers(arr[i])===last){count++; if(i>0 && isTiers(arr[i-1])===last)hit++}
    if(count>=5){const tr=hit/count;p=.75*p+.25*tr}
  }
  const edge=Math.abs(p-.5);
  const sample=Math.min(1,arr.length/100);
  const confScore=edge*2*sample;
  const confidence=confScore>=.32?"High":confScore>=.16?"Medium":"Low";
  return {tiers:Math.max(.01,Math.min(.99,p)),confidence,ready:true,note:"Estimate based on available historical evidence; not a guarantee."}
}
function renderSignals(){
  const a=history;
  const ints=intervals(a);
  const avg=ints.length?ints.reduce((x,y)=>x+y,0)/ints.length:null;
  const lastJump=a.length>1?wheelDistance(a[0],a[1]):null;
  const lastDir=a.length>1?direction(a[1],a[0]):"—";
  const items=[
    ["Frequency 5",rates(a.slice(0,5))==null?"—":pct(rates(a.slice(0,5)))],
    ["Frequency 20",rates(a.slice(0,20))==null?"—":pct(rates(a.slice(0,20)))],
    ["Frequency 50",rates(a.slice(0,50))==null?"—":pct(rates(a.slice(0,50)))],
    ["Frequency 100",rates(a.slice(0,100))==null?"—":pct(rates(a.slice(0,100)))],
    ["Frequency 250",rates(a.slice(0,250))==null?"—":pct(rates(a.slice(0,250)))],
    ["Sequence",a.length>1?(isTiers(a[0])===isTiers(a[1])?"Repeat":"Alternate"):"—"],
    ["Current interval",a.length?streak(a,false):0],
    ["Avg. interval",avg==null?"—":avg.toFixed(1)],
    ["Last jump",lastJump==null?"—":`${lastJump} pockets`],
    ["Direction",lastDir],
    ["Latest result",a.length?`${a[0]} · ${isTiers(a[0])?"Tiers":"No Tiers"}`:"—"],
    ["Data depth",a.length>=250?"250+":String(a.length)]
  ];
  $("signalsGrid").innerHTML=items.map(([k,v])=>`<div class="signal"><span>${k}</span><strong>${v}</strong></div>`).join("");
}
function render(){
  const a=history,m=model(a),r=rates(a);
  $("total").textContent=a.length;
  $("histRate").textContent=r==null?"—":pct(r);
  $("tiersStreak").textContent=streak(a,true);
  $("noTiersStreak").textContent=streak(a,false);
  const ints=intervals(a);
  $("interval").textContent=a.length&&isTiers(a[0])?0:streak(a,false);
  $("avgInterval").textContent=ints.length?(ints.reduce((x,y)=>x+y,0)/ints.length).toFixed(1):"—";
  $("status").textContent=m.ready?"MODEL ACTIVE":"DATA INSUFFICIENT";
  $("tiersPct").textContent=m.ready?pct(m.tiers):"—";
  $("noTiersPct").textContent=m.ready?pct(1-m.tiers):"—";
  $("mainPct").textContent=m.ready?pct(Math.max(m.tiers,1-m.tiers)):"—";
  $("mainSide").textContent=m.ready?(m.tiers>=.5?"Tiers":"No Tiers"):"Waiting for more results";
  $("confidence").textContent=m.confidence;
  $("tiersBar").style.width=m.ready?`${m.tiers*100}%`:"50%";
  $("modelNote").textContent=m.note;
  $("historyCount").textContent=`${Math.min(a.length,50)} / 50`;
  if(!a.length){$("history").className="history empty";$("history").textContent="No results yet."}
  else{
    $("history").className="history";
    $("history").innerHTML=a.slice(0,50).map((n,i)=>`<div class="history-row"><div class="number">${n}</div><div class="badge ${isTiers(n)?"tiers":"no-tiers"}">${isTiers(n)?"TIERS":"NO TIERS"}</div><div class="history-meta">${i===0?"LATEST":`${i} ago`}</div><div class="history-meta">${i<a.length-1?`${wheelDistance(n,a[i+1])} pockets`:"—"}</div></div>`).join("");
  }
  renderSignals();
}
function add(n){history.unshift(n);save();render()}
for(let n=0;n<=36;n++){const b=document.createElement("button");b.textContent=n;b.onclick=()=>add(n);$("keypad").appendChild(b)}
$("undoBtn").onclick=()=>{if(history.length){history.shift();save();render()}}
$("clearBtn").onclick=()=>{if(history.length&&confirm("Delete all stored results?")){history=[];save();render()}}
$("themeBtn").onclick=()=>document.body.classList.toggle("light");
render();
