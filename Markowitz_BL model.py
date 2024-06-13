import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def find_common_date_range(df):
    if df.index.name == '證券代碼':
        df = df.reset_index(drop=True)
    else:
        df = df.iloc[:, 1:]
    df = df.loc[:, (df != 0.00).all(axis=0)]
    df.columns = pd.to_datetime(df.columns)
    start_date = df.columns.min()
    end_date = df.columns.max()
    return start_date, end_date

def read_stock_data_from_excel(file_path):
    df = pd.read_excel(file_path, index_col='證券代碼')
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)
    return df

def get_price_data_from_excel(file_paths, tickers):
    combined_df = pd.DataFrame()
    for file_index, file_path in enumerate(file_paths):
        df = pd.read_excel(file_path)
        new_dates = pd.to_datetime(df.columns[2:])
        if combined_df.empty:
            combined_df = pd.DataFrame(index=new_dates)
        else:
            all_dates = combined_df.index.union(new_dates)
            combined_df = combined_df.reindex(all_dates)
        for ticker in tickers:
            close_price_df = df[(df['證券代碼'].str.startswith(ticker)) & (df['Data Field'] == '收盤價(元)')]
            if not close_price_df.empty:
                close_prices = close_price_df.drop(['證券代碼', 'Data Field'], axis=1).T
                close_prices.columns = [f'{ticker}_收盤價']
                close_prices.index = pd.to_datetime(close_prices.index)
                close_prices.index.name = '日期'
                for date in close_prices.index:
                    combined_df.loc[date, close_prices.columns] = close_prices.loc[date]
    return combined_df

def select_growth_and_value_stocks(df, start_date, end_date, top_n):
    # 將 DataFrame 的列標題轉換為字符串格式的日期
    df.columns = pd.to_datetime(df.columns, format='%Y/%m/%d').strftime('%Y-%m-%d')
    df.columns = df.columns.str.strip()  # 去除空格
    df.columns = df.columns.astype(str)
    # 將 start_date 和 end_date 轉換為無時間部分的日期格式
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')


    # 檢查 start_date 和 end_date 是否在 DataFrame 的列標題中
    if start_date not in df.columns:
        print(f"日期範圍錯誤：start_date {start_date} 不在 DataFrame 的列標題中")
    if end_date not in df.columns:
        print(f"日期範圍錯誤：end_date {end_date} 不在 DataFrame 的列標題中")

    # 選取指定日期範圍內的數據
    df_selected = df.loc[:, end_date:start_date]
    print("選取的 DataFrame：", df_selected)

    # 確認選取的 DataFrame 不為空
    if df_selected.empty:
        print("選取的 DataFrame 為空，請檢查日期範圍。")

    # 計算平均PB比率
    average_pb_ratios = df_selected.mean(axis=1)
    print(average_pb_ratios)

    # 按平均PB比率排序
    pb_sorted = average_pb_ratios.sort_values()

    # 選取最便宜和最貴的股票
    value_stocks = pb_sorted.head(top_n).index.tolist()
    growth_stocks = pb_sorted.tail(top_n).index.tolist()

    return growth_stocks, value_stocks

def calculate_daily_returns(data):
    return data.pct_change().dropna()

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std
    return returns, std, sharpe_ratio

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio
    return results, weights_record

def calculate_treynor_ratio(portfolio_return, risk_free_rate, beta):
    return (portfolio_return - risk_free_rate) / beta

def calculate_jensen_alpha(portfolio_return, risk_free_rate, market_return, beta):
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return portfolio_return - expected_return

def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df

def calculate_average_risk_free_rate(df):
    print("DataFrame的列名：", df.columns)
    df['年月日'] = pd.to_datetime(df['年月日'], format='%Y/%m/%d')
    df['年份'] = df['年月日'].dt.year
    avg_risk_free_rate_per_year = df.groupby('年份')['無風險利率'].mean()
    return avg_risk_free_rate_per_year
# Black-Litterman模型
def black_litterman(mean_returns, cov_matrix, risk_free_rate, P, Q, Omega):
    tau = 0.05  # 通常設置為一個小數值
    pi = mean_returns  # CAPM 預期收益
    M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(Omega) @ P)
    adj_mean_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    return adj_mean_returns

# 讀取Excel文件
file_path = 'D:\\Vscode\\vscode-python\\DQL_QL-MK_BL\\113stock\\analysis.xlsx'
risk_free_rate_df = read_excel_data(file_path)
avg_risk_free_rate_per_year = calculate_average_risk_free_rate(risk_free_rate_df)
print("每年平均無風險利率：")
print(avg_risk_free_rate_per_year)

# 獲取2014到2024年的無風險利率
risk_free_rates = avg_risk_free_rate_per_year.loc[2014:2024].to_dict()
print(f"2014到2024年的無風險利率: {risk_free_rates}")

file_paths = [
    'D:\\Vscode\\vscode-python\\DQL_QL-MK_BL\\113stock\\2014-2019 V2.xlsx',
    'D:\\Vscode\\vscode-python\\DQL_QL-MK_BL\\113stock\\2019-2024 V2.xlsx'
]
excel_file_path = 'D:\\Vscode\\vscode-python\\DQL_QL-MK_BL\\113stock\\pbratioV2.xlsx'
beta_file_path = 'D:\\Vscode\\vscode-python\\DQL_QL-MK_BL\\113stock\\capm_beta.xlsx'
stock_data_df = read_stock_data_from_excel(excel_file_path)
start_date, end_date = find_common_date_range(stock_data_df)
print(f"所有股票都有數據的時間區間 {start_date} 到 {end_date}")

# 選取成長股和價值股
growth_stocks, value_stocks = select_growth_and_value_stocks(stock_data_df, start_date, end_date, 15)
selected_tickers = growth_stocks + value_stocks

print("選中的股票:", selected_tickers)

price_data_df = get_price_data_from_excel(file_paths, selected_tickers)
print(price_data_df.head())

daily_returns = calculate_daily_returns(price_data_df)
mean_returns = daily_returns.mean()
print(mean_returns)

# 讀取beta值，只選取selected_tickers的股票
beta_df = pd.read_excel(beta_file_path, index_col='證券代碼')
# 處理無法轉換的值，保留負值
beta_df['CAPM_Beta 三年'] = pd.to_numeric(beta_df['CAPM_Beta 三年'], errors='coerce')

# 計算每支股票的CAPM_Beta 三年均值
beta_means = beta_df.groupby(beta_df.index)['CAPM_Beta 三年'].mean()
print(beta_means)

# 只選取selected_tickers的均值
selected_betas = beta_means.loc[selected_tickers].values

print("selected_betas:", selected_betas)
print("selected_betas 的數量:", len(selected_betas))

market_risk_premium = risk_free_rate_df['市場風險溢酬'].mean().astype(float)
print("market_risk_premium:", market_risk_premium)
cov_matrix = daily_returns.cov()
num_portfolios = 10000
avg_risk_free_rate = np.mean([risk_free_rates[year] for year in range(start_date.year, end_date.year + 1)]) / 100

# 計算CAPM預期收益率
capm_returns = avg_risk_free_rate + selected_betas * market_risk_premium
print("CAPM預期收益率", capm_returns)
print("平均預期收益率", mean_returns)
# 檢查協方差矩陣和均值返回的維度是否一致
print("協方差矩陣維度:", cov_matrix.shape)
print("均值返回維度:", mean_returns.shape)

# 確保協方差矩陣的維度和均值返回的維度一致
if cov_matrix.shape[0] != len(mean_returns):
    cov_matrix = cov_matrix.loc[selected_tickers, selected_tickers]

# Black-Litterman模型的參數
P = np.eye(len(selected_tickers))  # 假設每個股票有一個單獨的觀點
print("P矩陣:", P)
Q = capm_returns  # 使用CAPM預期收益作為觀點的預期收益
Omega = np.diag(np.ones(len(selected_tickers)) * 0.05)  # 設置觀點的不確定性
print("Omega矩陣:", Omega)

# 調整後的預期收益
adjusted_mean_returns_bl = black_litterman(mean_returns, cov_matrix, avg_risk_free_rate, P, Q, Omega)
print("BL模型調整後的預期收益:", adjusted_mean_returns_bl)

# 使用Black-Litterman模型調整後的預期收益生成投資組合
results_bl, weights_bl = generate_random_portfolios(num_portfolios, adjusted_mean_returns_bl, cov_matrix, avg_risk_free_rate)

max_sharpe_idx_bl = np.argmax(results_bl[2])
max_sharpe_portfolio_bl = weights_bl[max_sharpe_idx_bl]

max_sharpe_return_bl, max_sharpe_std_bl, max_sharpe_ratio_bl = portfolio_performance(max_sharpe_portfolio_bl, adjusted_mean_returns_bl, cov_matrix, avg_risk_free_rate)
print("BL模型最大夏普比率的投資組合:")
formatted_weights_bl = ["{:.8f}".format(weight) for weight in max_sharpe_portfolio_bl]
print("權重:", formatted_weights_bl)
print("預期年化收益率:", max_sharpe_return_bl)
print("預期年化風險:", max_sharpe_std_bl)
print(f"夏普比率: {max_sharpe_ratio_bl}")

treynor_ratio_bl = calculate_treynor_ratio(max_sharpe_return_bl, avg_risk_free_rate, 1.0)
jensen_alpha_bl = calculate_jensen_alpha(max_sharpe_return_bl, avg_risk_free_rate, 0.08, 1.0)
print(f"崔娜指标: {treynor_ratio_bl}")
print(f"詹森指标: {jensen_alpha_bl}")

plt.figure(figsize=(10, 6))
plt.scatter(results_bl[1, :], results_bl[0, :], c=results_bl[2, :], cmap='YlGnBu', marker='o')
plt.title('BL模型有效前沿')
plt.xlabel('投資組合風險')
plt.ylabel('投資組合收益率')
plt.colorbar(label='夏普比率')
plt.show()

# Markowitz模型
results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, avg_risk_free_rate)

max_sharpe_idx = np.argmax(results[2])
max_sharpe_portfolio = weights[max_sharpe_idx]

max_sharpe_return, max_sharpe_std, _ = portfolio_performance(max_sharpe_portfolio, mean_returns, cov_matrix, avg_risk_free_rate)
print("MK模型預期收益:", mean_returns)
print("最大夏普比率的投資組合:")
formatted_weights = ["{:.8f}".format(weight) for weight in max_sharpe_portfolio]
print("權重:", formatted_weights)
print("預期年化收益率:", max_sharpe_return)
print("預期年化風險:", max_sharpe_std)
treynor_ratio = calculate_treynor_ratio(max_sharpe_return, avg_risk_free_rate, 1.0)
jensen_alpha = calculate_jensen_alpha(max_sharpe_return, avg_risk_free_rate, 0.08, 1.0)
print(f"夏普比率: {results[2, max_sharpe_idx]}")
print(f"崔娜指标: {treynor_ratio}")
print(f"詹森指标: {jensen_alpha}")

plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
plt.title('有效前沿')
plt.xlabel('投資組合風險')
plt.ylabel('投資組合收益率')
plt.colorbar(label='夏普比率')
plt.show()
