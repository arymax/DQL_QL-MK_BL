import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

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
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df.loc[:, start_date:end_date]
    average_pb_ratios = df.mean(axis=1)
    pb_sorted = average_pb_ratios.sort_values()
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

def black_litterman(mean_returns, cov_matrix, risk_free_rate, P, Q, Omega):
    tau = 0.05  # 通常設置為一個小數值

    pi = mean_returns  # CAPM 預期收益
    M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(Omega) @ P)
    adj_mean_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    return adj_mean_returns

# Trading environment using gym
class TradingEnv(gym.Env):
    def __init__(self, initial_portfolio, data, mean_returns, cov_matrix, risk_free_rate):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_portfolio = initial_portfolio
        self.current_portfolio = np.array(initial_portfolio, dtype=np.float32)
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.current_step = 0
        self.num_assets = len(initial_portfolio)
        self.risk_free_rate = risk_free_rate

        self.action_space = spaces.Discrete(self.num_assets * 3)  # 將每個資產的動作（買進、持有、賣出）展開
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_assets + len(data.columns),), dtype=np.float32)

        # 初始化最佳夏普比率和對應的投資組合
        self.best_sharpe_ratio = -np.inf
        self.best_portfolio_weights = None
        self.best_annual_return = None

    def reset(self):
        self.current_step = 0
        self.current_portfolio = np.array(self.initial_portfolio, dtype=np.float32)
        return self.get_state()

    def step(self, action_code):
        # 解碼動作
        asset_index = action_code // 3
        action_type = action_code % 3

        # 確保資產索引在範圍內
        if asset_index < 0 or asset_index >= self.num_assets:
            raise ValueError(f"Asset index {asset_index} is out of bounds")

        # 根據動作類型更新投資組合
        if action_type == 0:  # 賣出
            self.current_portfolio[asset_index] = max(self.current_portfolio[asset_index] - 0.01, 0)
        elif action_type == 2:  # 買入
            self.current_portfolio[asset_index] = min(self.current_portfolio[asset_index] + 0.01, 1)
        # 持有動作 (action_type == 1) 不需要額外處理

        # 再平衡組合，確保總和為1
        self.current_portfolio = self.current_portfolio / np.sum(self.current_portfolio)

        # 進入下一步
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0
            done = True
        else:
            done = False

        # 計算投資組合表現
        portfolio_return, portfolio_std, sharpe_ratio = portfolio_performance(
            self.current_portfolio, self.mean_returns, self.cov_matrix, self.risk_free_rate
        )

        # 記錄最佳夏普比率和對應的投資組合
        if sharpe_ratio > self.best_sharpe_ratio:
            self.best_sharpe_ratio = sharpe_ratio
            self.best_portfolio_weights = self.current_portfolio.copy()
            self.best_annual_return = portfolio_return

        # 設定獎勵為夏普比率
        reward = sharpe_ratio

        state = self.get_state()
        return state, reward, done, {"portfolio_return": portfolio_return, "portfolio_std": portfolio_std, "sharpe_ratio": sharpe_ratio}

    def get_state(self):
        current_prices = self.data.iloc[self.current_step].values
        state = np.concatenate([self.current_portfolio, current_prices])
        return state / np.max(state)

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
# Markowitz模型
results_mk, weights_mk = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, avg_risk_free_rate)
max_sharpe_idx = np.argmax(results_mk[2])
max_sharpe_portfolio = weights_mk[max_sharpe_idx]
max_sharpe_return, max_sharpe_std, _ = portfolio_performance(max_sharpe_portfolio, mean_returns, cov_matrix, avg_risk_free_rate)

# 初始化Q-Learning
initial_weights = max_sharpe_portfolio_bl
env = TradingEnv(
    initial_portfolio=initial_weights,
    data=price_data_df,
    mean_returns=adjusted_mean_returns_bl,
    cov_matrix=cov_matrix,
    risk_free_rate=avg_risk_free_rate
)

q_table = {}
total_rewards = []
average_sharpe_ratios = []
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.01
total_episodes = 20

best_sharpe_ratio = -np.inf
best_portfolio = None
best_weights = None
best_return = None
best_std = None

# 使用隨機生成的權重作為初始策略進行Q-Learning
for episode in range(total_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    sharpe_ratios = []
    print(f"Episode {episode + 1}/{total_episodes}")

    while not done:
        state_key = str(state)
        if state_key not in q_table:
            q_table[state_key] = np.zeros(env.action_space.n)

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
            print(f"Random action: {action}")
        else:
            action = np.argmax(q_table[state_key])
            print(f"Greedy action: {action}")

        new_state, reward, done, info = env.step(action)
        sharpe_ratios.append(info['sharpe_ratio'])

        new_state_key = str(new_state)
        if new_state_key not in q_table:
            q_table[new_state_key] = np.zeros(env.action_space.n)

        q_table[state_key][action] = (1 - learning_rate) * q_table[state_key][action] + \
                                     learning_rate * (reward + gamma * np.max(q_table[new_state_key]))

        state = new_state
        total_reward += reward

        if reward > best_sharpe_ratio:
            best_sharpe_ratio = reward
            best_portfolio = env.current_portfolio
            best_weights = env.current_portfolio
            best_return = info["portfolio_return"]
            best_std = info["portfolio_std"]

    average_sharpe = np.mean(sharpe_ratios)
    average_sharpe_ratios.append(average_sharpe)
    print(f"Average Sharpe Ratio for episode {episode + 1}: {average_sharpe}")
    print(f"Total reward for episode {episode + 1}: {total_reward}")
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
    total_rewards.append(total_reward)

average_reward = sum(total_rewards) / total_episodes
average_sharpe_ratio = sum(average_sharpe_ratios) / total_episodes
print(f"Average Reward: {average_reward}")
print(f"Average Sharpe Ratio: {average_sharpe_ratio}")
print(f"Best Sharpe Ratio: {best_sharpe_ratio}")
print(f"Best Portfolio Weights: {best_weights}")
print(f"Expected Annual Return: {best_return}")
print(f"Expected Annual Volatility: {best_std}")

# 計算Treynor Ratio和Jensen Alpha
market_return = 0.08  # 假設市場年化收益率為8%
beta = 1.0  # 假設投資組合的Beta為1
treynor_ratio = calculate_treynor_ratio(best_return, avg_risk_free_rate, beta)
jensen_alpha = calculate_jensen_alpha(best_return, avg_risk_free_rate, market_return, beta)
print(f"Treynor Ratio: {treynor_ratio}")
print(f"Jensen Alpha: {jensen_alpha}")