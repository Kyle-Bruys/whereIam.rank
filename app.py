import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import beta

# 페이지 기본 설정
st.set_page_config(page_title="의과대학 성적 정밀 추정기", layout="wide")

st.title("📊 의과대학 시험 등수 정밀 추정 대시보드")
st.markdown("""
임의의 가정을 모두 배제하고, **평균 점수를 제약 조건으로 한 베타 분포 최적화(Quantile Matching Estimation)**를 통해 가장 수리적으로 강건한(Robust) 예상 등수를 계산합니다. 
데이터가 스스로 분산을 증명하여 평균의 백분위 위치를 자동으로 찾아냅니다.
""")

# --- 1. 사이드바: 데이터 입력 ---
with st.sidebar:
    st.header("📝 시험 데이터 입력")
    total_q = st.number_input("총 문제 수", min_value=10, max_value=500, value=120)
    
    # 수정됨: value가 total_q를 넘지 않도록 min() 함수 적용
    mean_score = st.number_input("전체 평균 (맞은 개수)", min_value=0.0, max_value=float(total_q), value=min(82.0, float(total_q)))
    
    st.markdown("---")
    st.markdown("**하위권 커트라인 데이터** (분산 추정용)")
    
    # 수정됨: 이하 항목들도 모두 value에 min() 함수 적용
    cut_10 = st.number_input("하위 10% 점수", min_value=0, max_value=int(total_q), value=min(60, int(total_q)))
    cut_20 = st.number_input("하위 20% 점수", min_value=0, max_value=int(total_q), value=min(68, int(total_q)))
    cut_30 = st.number_input("하위 30% 점수", min_value=0, max_value=int(total_q), value=min(74, int(total_q)))
    
    st.markdown("---")
    my_score = st.number_input("🎯 내 맞은 개수", min_value=0.0, max_value=float(total_q), value=min(90.0, float(total_q)))

# --- 2. 데이터 정규화 ---
mu_n = mean_score / total_q
c_10_n = cut_10 / total_q
c_20_n = cut_20 / total_q
c_30_n = cut_30 / total_q
my_norm = my_score / total_q

# --- 3. 1차원 수학적 최적화 (목적 함수) ---
def loss_function(alpha_param):
    # E[X] = alpha / (alpha + beta) 수식을 이용하여 beta를 alpha에 종속시킴
    beta_param = alpha_param * (1 - mu_n) / mu_n
    
    # 각 커트라인의 이론적 CDF와 실제 확률(0.1, 0.2, 0.3) 간의 오차 제곱합
    err_10 = (beta.cdf(c_10_n, alpha_param, beta_param) - 0.10)**2
    err_20 = (beta.cdf(c_20_n, alpha_param, beta_param) - 0.20)**2
    err_30 = (beta.cdf(c_30_n, alpha_param, beta_param) - 0.30)**2
    
    return err_10 + err_20 + err_30

# 최적의 alpha 값 탐색 (0.01 ~ 100.0 범위 내)
res = minimize_scalar(loss_function, bounds=(0.01, 100.0), method='bounded')

if res.success:
    # 최적 파라미터 도출
    alpha_opt = res.x
    beta_opt = alpha_opt * (1 - mu_n) / mu_n
    
    # 결과 계산
    my_percentile = beta.cdf(my_norm, alpha_opt, beta_opt)
    my_top_rank = (1 - my_percentile) * 100
    
    mean_percentile = beta.cdf(mu_n, alpha_opt, beta_opt)
    mean_top_rank = (1 - mean_percentile) * 100

    # --- 4. 메인 화면: 결과 요약 ---
    st.subheader("🏆 추정 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("통계 파라미터 (α / β)", f"{alpha_opt:.2f} / {beta_opt:.2f}")
    
    # 이 대시보드의 핵심: 동역학적으로 계산된 평균의 실제 등수
    col2.metric("💡 계산된 평균의 실제 등수", f"상위 {mean_top_rank:.1f}%", help=f"하위 {mean_percentile*100:.1f}% (분산을 통해 역산된 위치)")
    
    # 최종 등수
    col3.metric("🎯 나의 예상 등수", f"상위 {my_top_rank:.1f}%", delta="안전권" if my_top_rank <= 70 else "위험권 (하위 30% 이하)", delta_color="inverse")

    # --- 5. 시각화 (Matplotlib) ---
    st.markdown("---")
    st.subheader("📈 정밀 누적 분포 곡선 (Mathematical Beta CDF)")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 부드러운 곡선을 그리기 위한 X축 데이터
    x_vals = np.linspace(0, total_q, 500)
    y_vals = beta.cdf(x_vals / total_q, alpha_opt, beta_opt) * 100
    
    # 분포 곡선
    ax.plot(x_vals, y_vals, label='Optimized Beta Curve', linewidth=3, color='royalblue')
    
    # 제공된 닻(Anchor) 데이터 마커
    ax.scatter([cut_10, cut_20, cut_30], [10, 20, 30], color='black', s=60, zorder=5, label='Known Cutoffs (10%, 20%, 30%)')
    
    # 동적으로 계산된 평균 점수의 위치 마커
    ax.scatter([mean_score], [mean_percentile*100], color='green', s=100, zorder=6, label=f'Calculated Mean Position ({mean_percentile*100:.1f}%)')
    
    # 내 점수 위치 수직선 및 수평선
    ax.axvline(x=my_score, color='red', linestyle='--', linewidth=2, label=f'My Score ({my_score})')
    ax.axhline(y=my_percentile*100, color='red', linestyle='--', alpha=0.3)
    
    # 그래프 꾸미기
    ax.set_xlabel("Score (Correct Answers)", fontsize=12)
    ax.set_ylabel("Cumulative Percentile (Bottom %)", fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    st.pyplot(fig)
    
else:
    st.error("최적의 분포를 찾는 데 실패했습니다. 입력된 하위권 커트라인 수치가 수학적으로 불가능한 형태(예: 점수가 역전됨)일 수 있습니다.")