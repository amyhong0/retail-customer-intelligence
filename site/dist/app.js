const metrics={
  ndcg:{label:'NDCG@10',baseline:0.0396,candidate:0.1227,help:'NDCG@10: 실제 구매한 상품이 추천 10개 중 높은 순위에 있는지 평가합니다. 값이 클수록 좋습니다.',description:'인기 상품 추천 대비 NDCG@10의 상대 증가율입니다. 실제 구매한 상품을 더 높은 순위에 추천했습니다.'},
  recall:{label:'Recall@10',baseline:0.0367,candidate:0.0834,help:'Recall@10: 실제 구매한 상품 중 추천 10개에 포함된 비율입니다. 고객별 비율의 평균이며, 값이 클수록 좋습니다.',description:'인기 상품 추천 대비 Recall@10의 상대 증가율입니다. 실제 구매 상품을 추천 목록에 더 많이 포함했습니다.'},
  precision:{label:'Precision@10',baseline:0.0070,candidate:0.0714,help:'Precision@10: 추천 10개 중 실제 구매한 상품의 비율입니다. 고객별 비율의 평균이며, 값이 클수록 좋습니다.',description:'인기 상품 추천 대비 Precision@10의 상대 증가율입니다. 추천한 상품이 실제 구매와 더 자주 일치했습니다.'}
};
const tabs=[...document.querySelectorAll('[data-metric]')];
function renderMetric(key){
  const metric=metrics[key]; if(!metric)return;
  tabs.forEach(tab=>{const selected=tab.dataset.metric===key;tab.classList.toggle('active',selected);tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1});
  document.getElementById('baseline-bar').style.width=`${metric.baseline/0.2*100}%`;
  document.getElementById('candidate-bar').style.width=`${metric.candidate/0.2*100}%`;
  document.getElementById('baseline-value').textContent=metric.baseline.toFixed(4);
  document.getElementById('candidate-value').textContent=metric.candidate.toFixed(4);
  document.getElementById('metric-delta').textContent=`+${((metric.candidate/metric.baseline-1)*100).toFixed(1)}%`;
  document.getElementById('metric-description').textContent=metric.description;
  document.getElementById('metric-help').textContent=metric.help;
  document.querySelector('.chart-comparison').setAttribute('aria-label',`인기 상품 추천과 상품 협업 필터링의 ${metric.label} 비교: ${metric.baseline.toFixed(4)} 대 ${metric.candidate.toFixed(4)}`);
}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>renderMetric(tab.dataset.metric));
  tab.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();
    const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    tabs[next].focus(); renderMetric(tabs[next].dataset.metric);
  });
});
renderMetric('ndcg');
