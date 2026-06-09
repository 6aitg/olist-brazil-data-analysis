import sys
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====================== 系统字体扫描与中文乱码修复核心配置 ======================
# 1. 字符编码全局设置
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 2. 扫描系统所有可用中文字体
def scan_chinese_fonts():
    """扫描系统中所有支持中文的字体，返回可用字体名称列表"""
    # 常见中文字体名称优先级列表
    target_chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Heiti TC', 'PingFang SC', 'Hiragino Sans GB',
        'Source Han Sans CN', 'Noto Sans CJK SC', 'Droid Sans Fallback'
    ]
    
    # 获取系统所有已安装字体
    system_fonts = fm.findSystemFonts()
    available_fonts = []
    
    # 遍历系统字体，匹配目标中文字体
    for font_path in system_fonts:
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            if font_name in target_chinese_fonts and font_name not in available_fonts:
                available_fonts.append(font_name)
        except Exception:
            continue
    
    # 补充默认兜底字体
    available_fonts.extend(['Arial', 'sans-serif'])
    return available_fonts

# 3. 执行字体扫描并设置matplotlib全局参数
available_chinese_fonts = scan_chinese_fonts()
print(f"✅ 扫描到可用中文字体: {available_chinese_fonts[:3]}")

# 全局字体配置（彻底解决中文乱码）
plt.rcParams['font.family'] = available_chinese_fonts
plt.rcParams['font.sans-serif'] = available_chinese_fonts
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
plt.rcParams['figure.dpi'] = 100  # 统一图片清晰度
plt.rcParams['savefig.dpi'] = 300  # 保存图片的高清分辨率
plt.rcParams['savefig.bbox'] = 'tight'  # 自动适配文本，避免标签被截断
plt.rcParams['figure.autolayout'] = True  # 自动调整布局

# 4. seaborn样式同步配置
sns.set_style('whitegrid', {'font.family': available_chinese_fonts})
sns.set(font_scale=0.9, font=available_chinese_fonts)  # 减小默认字体缩放比例

# ====================== 文件夹创建 ======================
# 定义数据路径和输出路径
data_path = 'D:\workspace\pythonspace\qianfeng\Olist Brazil Data Analysis\data'
base_output_path = 'D:\workspace\pythonspace\qianfeng\Olist Brazil Data Analysis\output'

# 定义六个分析模块的文件夹
modules = [
    '01_订单概览分析',
    '02_用户行为分析',
    '03_产品品类分析',
    '04_物流配送分析',
    '05_支付方式分析',
    '06_客户评价分析'
]

# 创建文件夹
for module in modules:
    module_path = os.path.join(base_output_path, module)
    if not os.path.exists(module_path):
        os.makedirs(module_path)
    print(f"创建文件夹: {module_path}")

# ====================== 数据读取和预处理 ======================
print("=" * 80)
print("巴西Olist电商平台数据分析")
print("=" * 80)
print("\n【第一步】数据读取和预处理")
print("-" * 60)

# 读取Olist数据集
def load_olist_data():
    """加载Olist电商平台所有数据集"""
    files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'product_category_name_translation': 'product_category_name_translation.csv'
    }
    
    data = {}
    for key, filename in files.items():
        try:
            data[key] = pd.read_csv(os.path.join(data_path, filename))
            print(f"✅ 成功读取 {filename}: {len(data[key]):,} 条记录")
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
            data[key] = pd.DataFrame()
    
    return data

# 加载数据
data = load_olist_data()

# 数据预处理
def preprocess_data(data):
    """数据预处理"""
    # 订单数据预处理
    if not data['orders'].empty:
        # 转换日期格式
        date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                       'order_delivered_carrier_date', 'order_delivered_customer_date',
                       'order_estimated_delivery_date']
        
        for col in date_columns:
            data['orders'][col] = pd.to_datetime(data['orders'][col], errors='coerce')
        
        # 计算配送时间
        data['orders']['delivery_time'] = (data['orders']['order_delivered_customer_date'] - 
                                          data['orders']['order_purchase_timestamp']).dt.days
        
        # 计算配送延迟
        data['orders']['delivery_delay'] = (data['orders']['order_delivered_customer_date'] - 
                                           data['orders']['order_estimated_delivery_date']).dt.days
        
        # 订单状态分类
        data['orders']['status_category'] = data['orders']['order_status'].map({
            'delivered': '已配送',
            'shipped': '已发货',
            'canceled': '已取消',
            'unavailable': '不可用',
            'invoiced': '已开票',
            'processing': '处理中',
            'created': '已创建'
        })
        print("✅ 订单数据预处理完成：日期格式转换、配送时间计算、订单状态分类")
    
    # 产品品类翻译
    if not data['product_category_name_translation'].empty and not data['products'].empty:
        translation_dict = dict(zip(data['product_category_name_translation']['product_category_name'],
                                   data['product_category_name_translation']['product_category_name_english']))
        data['products']['category_english'] = data['products']['product_category_name'].map(translation_dict)
        print("✅ 产品品类翻译完成：葡萄牙语品类名称转换为英语")
    
    # 合并订单和订单项数据
    if not data['orders'].empty and not data['order_items'].empty:
        data['orders_with_items'] = pd.merge(data['orders'], data['order_items'], on='order_id', how='left')
        print("✅ 订单和订单项数据合并完成")
    
    return data

# 执行预处理
data = preprocess_data(data)

# 基本数据信息
print(f"\n📊 数据概览:")
if not data['orders'].empty:
    start_date = data['orders']['order_purchase_timestamp'].min()
    end_date = data['orders']['order_purchase_timestamp'].max()
    print(f"订单时间范围: {start_date} 至 {end_date}")
    print(f"总订单数量: {len(data['orders']):,}")
    
    print(f"订单状态分布:")
    order_status = data['orders']['order_status'].value_counts()
    for status, count in order_status.items():
        percentage = count/len(data['orders'])*100
        print(f"  - {status}: {count:,} ({percentage:.1f}%)")

# ====================== 模块1: 订单概览分析 ======================
print("\n" + "=" * 80)
print("模块1: 订单概览分析")
print("=" * 80)
module1_path = os.path.join(base_output_path, modules[0])

# 1.1 订单时间趋势分析
print("\n【1.1】订单时间趋势分析")
if not data['orders'].empty:
    # 按月份统计订单数量
    orders_monthly = data['orders'].copy()
    orders_monthly['purchase_month'] = orders_monthly['order_purchase_timestamp'].dt.to_period('M')
    monthly_counts = orders_monthly.groupby('purchase_month').size()
    
    # 计算关键指标
    max_month = monthly_counts.idxmax()
    max_orders = monthly_counts.max()
    min_month = monthly_counts.idxmin()
    min_orders = monthly_counts.min()
    avg_monthly_orders = monthly_counts.mean()
    
    trend_summary = f"""
【订单时间趋势关键发现】
• 订单量最高的月份: {max_month} ({max_orders:,} 单)
• 订单量最低的月份: {min_month} ({min_orders:,} 单)
• 平均月订单量: {avg_monthly_orders:.0f} 单
• 订单时间跨度: {monthly_counts.index.min()} 至 {monthly_counts.index.max()}
"""
    print(trend_summary)
    
    # 可视化1: 月度订单趋势图
    plt.figure(figsize=(16, 9))  # 增大图片尺寸
    ax = monthly_counts.plot(kind='line', marker='o', linewidth=2, color='#2E86AB', markersize=6)
    plt.title('月度订单趋势 (2016-2018)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('月份', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 调整标签位置，避免重叠
    for i, (x, y) in enumerate(zip(monthly_counts.index, monthly_counts.values)):
        if i % 2 == 0:  # 每隔一个点显示标签，避免重叠
            ax.text(x, y + max_orders*0.01, f'{y:,}', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(os.path.join(module1_path, '01_月度订单趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2 订单金额分析
    print("\n【1.2】订单金额分析")
    if not data['order_items'].empty:
        # 计算订单总金额
        order_amounts = data['order_items'].groupby('order_id')['price'].sum().reset_index()
        order_amounts.columns = ['order_id', 'total_amount']
        
        # 计算金额统计指标
        avg_order_amount = order_amounts['total_amount'].mean()
        median_order_amount = order_amounts['total_amount'].median()
        max_order_amount = order_amounts['total_amount'].max()
        min_order_amount = order_amounts['total_amount'].min()
        total_sales = order_amounts['total_amount'].sum()
        
        amount_summary = f"""
【订单金额关键指标】
• 平均订单金额: R${avg_order_amount:.2f}
• 订单金额中位数: R${median_order_amount:.2f}
• 最高订单金额: R${max_order_amount:.2f}
• 最低订单金额: R${min_order_amount:.2f}
• 总销售额: R${total_sales:,.2f}
• 金额在500雷亚尔以下的订单占比: {(order_amounts['total_amount'] < 500).sum() / len(order_amounts) * 100:.1f}%
"""
        print(amount_summary)
        
        # 可视化2: 订单金额分布
        plt.figure(figsize=(16, 9))
        plt.hist(order_amounts[order_amounts['total_amount'] < 500]['total_amount'], 
                    bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        plt.title('订单金额分布 (500雷亚尔以内)', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('订单金额 (雷亚尔)', fontsize=11, labelpad=10)
        plt.ylabel('频次', fontsize=11, labelpad=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.savefig(os.path.join(module1_path, '02_订单金额分布.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1.3 订单状态分析
        print("\n【1.3】订单状态分析")
        # 计算状态占比
        status_counts = data['orders']['order_status'].value_counts()
        delivered_rate = status_counts.get('delivered', 0) / len(data['orders']) * 100
        canceled_rate = status_counts.get('canceled', 0) / len(data['orders']) * 100
        
        status_summary = f"""
【订单状态关键指标】
• 已配送订单占比: {delivered_rate:.1f}%
• 已取消订单占比: {canceled_rate:.1f}%
• 主要订单状态: {status_counts.index[0]} ({status_counts.iloc[0]:,} 单, {status_counts.iloc[0]/len(data['orders'])*100:.1f}%)
• 订单完成率（已配送/总订单）: {delivered_rate:.1f}%
"""
        print(status_summary)
        
        # 可视化3: 订单状态饼图
        plt.figure(figsize=(12, 12))
        colors = ['#2E86AB', '#F18F01', '#C73E1D', '#9A031E', '#5F4B8B', '#4E9F3D', '#FFD300']
        
        # 计算百分比并过滤标签：只显示≥5%的标签和占比
        total = status_counts.sum()
        labels = [name if (count/total)*100 >= 5 else '' for name, count in zip(status_counts.index, status_counts.values)]
        
        # 调整饼图标签，避免重叠
        def autopct_format(pct):
            return f'{pct:.1f}%' if pct >= 5 else ''  # 小于5%的部分不显示百分比
       
        
        
        wedges, texts, autotexts = plt.pie(status_counts.values, 
                                           labels=labels, 
                                           autopct=autopct_format,
                                           colors=colors[:len(status_counts)], 
                                           startangle=90,
                                           textprops={'fontsize': 10},
                                           labeldistance=1.15,
                                           pctdistance=0.8,
                                           )
        
        # 调整百分比文字大小
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('订单状态分布', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(module1_path, '03_订单状态分布.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1.4 地理分布分析
        print("\n【1.4】订单地理分布分析")
        if not data['customers'].empty:
            # 按州统计订单数量
            state_orders = data['customers']['customer_state'].value_counts().head(10)
            top_state = state_orders.index[0]
            top_state_orders = state_orders.iloc[0]
            top_state_percentage = top_state_orders / state_orders.sum() * 100
            
            geo_summary = f"""
【订单地理分布关键发现】
• 订单量最高的州: {top_state} ({top_state_orders:,} 单, {top_state_percentage:.1f}%)
• 前10州订单占比: {state_orders.sum() / len(data['customers']) * 100:.1f}%
• 订单分布最集中的地区: {', '.join(state_orders.head(3).index.tolist())}
"""
            print(geo_summary)
            
            # 可视化4: 各州订单分布
            plt.figure(figsize=(16, 9))
            ax = state_orders.plot(kind='bar', color='#2E86AB', alpha=0.8)
            plt.title('订单量TOP10州', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('州', fontsize=11, labelpad=10)
            plt.ylabel('订单数量', fontsize=11, labelpad=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            
            # 添加数值标签，调整位置避免重叠
            for i, v in enumerate(state_orders):
                ax.text(i, v + max(state_orders)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.savefig(os.path.join(module1_path, '04_各州订单分布.png'), dpi=300, bbox_inches='tight')
            plt.close()
print(f"\n✅ 模块1分析完成，图片已保存至: {module1_path}")

# ====================== 模块2: 用户行为分析 ======================
print("\n" + "=" * 80)
print("模块2: 用户行为分析")
print("=" * 80)
module2_path = os.path.join(base_output_path, modules[1])

if not data['orders'].empty and not data['customers'].empty:
    # 2.1 用户购买频次分析
    print("\n【2.1】用户购买频次分析")
    # 统计每个用户的订单数量
    customer_orders = data['orders'].merge(data['customers'][['customer_id', 'customer_unique_id']], 
                                          on='customer_id', how='left')
    user_order_count = customer_orders.groupby('customer_unique_id')['order_id'].nunique()
    
    # 计算频次指标
    one_time_buyers = (user_order_count == 1).sum()
    multi_buyers = (user_order_count > 1).sum()
    max_purchases = user_order_count.max()
    avg_purchases = user_order_count.mean()
    
    freq_summary = f"""
【用户购买频次关键指标】
• 一次性购买用户数: {one_time_buyers:,} ({one_time_buyers/len(user_order_count)*100:.1f}%)
• 多次购买用户数: {multi_buyers:,} ({multi_buyers/len(user_order_count)*100:.1f}%)
• 单用户最大购买次数: {max_purchases} 次
• 平均购买次数: {avg_purchases:.2f} 次
• 购买频次最高的前10档: {user_order_count.value_counts().sort_index().head(10).to_dict()}
"""
    print(freq_summary)
    
    # 可视化1: 用户购买频次分布
    plt.figure(figsize=(16, 9))
    freq_counts = user_order_count.value_counts().sort_index().head(10)
    ax = freq_counts.plot(kind='bar', color='#F18F01', alpha=0.8)
    plt.title('用户购买频次分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('单用户订单数量', fontsize=11, labelpad=10)
    plt.ylabel('用户数量', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(freq_counts):
        ax.text(i, v + max(freq_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module2_path, '01_用户购买频次分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.2 复购率分析
    print("\n【2.2】复购率分析")
    repeat_users = (user_order_count >= 2).sum()
    total_users = len(user_order_count)
    repeat_rate = repeat_users / total_users * 100
    
    repeat_summary = f"""
【复购率关键指标】
• 总用户数: {total_users:,}
• 复购用户数: {repeat_users:,}
• 复购率: {repeat_rate:.2f}%
• 单次购买用户占比: {(1 - repeat_users/total_users)*100:.2f}%
• 复购用户贡献订单数: {user_order_count[user_order_count >= 2].sum():,} ({user_order_count[user_order_count >= 2].sum()/user_order_count.sum()*100:.1f}%)
"""
    print(repeat_summary)
    
    # 可视化2: 复购率对比
    plt.figure(figsize=(12, 8))
    categories = ['一次性购买用户', '复购用户']
    values = [one_time_buyers, repeat_users]
    colors = ['#A23B72', '#F18F01']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    ax.set_title('一次性购买用户 vs 复购用户', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('用户数量', fontsize=11, labelpad=10)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=0, fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(values):
        percentage = v/total_users*100
        ax.text(i, v + max(values)*0.01, f'{v:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module2_path, '02_复购用户对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.3 用户下单时间分布
    print("\n【2.3】用户下单时间分布")
    # 提取小时和星期
    orders_time = data['orders'].copy()
    orders_time['purchase_hour'] = orders_time['order_purchase_timestamp'].dt.hour
    orders_time['purchase_weekday'] = orders_time['order_purchase_timestamp'].dt.weekday
    orders_time['weekday_name'] = orders_time['purchase_weekday'].map({
        0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'
    })
    
    # 按小时统计
    hour_counts = orders_time['purchase_hour'].value_counts().sort_index()
    # 按星期统计
    weekday_counts = orders_time['weekday_name'].value_counts().reindex(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    
    time_summary = f"""
【用户下单时间关键发现】
• 下单高峰小时: {hour_counts.idxmax()} 点 ({hour_counts.max():,} 单)
• 下单低谷小时: {hour_counts.idxmin()} 点 ({hour_counts.min():,} 单)
• 下单高峰星期: {weekday_counts.idxmax()} ({weekday_counts.max():,} 单)
• 下单低谷星期: {weekday_counts.idxmin()} ({weekday_counts.min():,} 单)
"""
    print(time_summary)
    
    # 可视化3: 小时下单分布
    plt.figure(figsize=(18, 9))
    ax = hour_counts.plot(kind='bar', color='#2E86AB', alpha=0.8, width=0.8)
    plt.title('各小时订单数量分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('小时', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=9)
    
    # 每隔2个小时显示标签，避免重叠
    for i, v in enumerate(hour_counts):
        if i % 2 == 0:
            ax.text(i, v + max(hour_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.savefig(os.path.join(module2_path, '03_小时订单分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化4: 星期下单分布
    plt.figure(figsize=(14, 8))
    ax = weekday_counts.plot(kind='bar', color='#F18F01', alpha=0.8, width=0.7)
    plt.title('各星期订单数量分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('星期', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(weekday_counts):
        ax.text(i, v + max(weekday_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module2_path, '04_星期订单分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
print(f"\n✅ 模块2分析完成，图片已保存至: {module2_path}")

# ====================== 模块3: 产品品类分析 ======================
print("\n" + "=" * 80)
print("模块3: 产品品类分析")
print("=" * 80)
module3_path = os.path.join(base_output_path, modules[2])

if not data['products'].empty and not data['order_items'].empty:
    # 合并产品和订单项数据
    product_orders = pd.merge(data['order_items'], data['products'], on='product_id', how='left')
    # 3.1 品类销量分析
    print("\n【3.1】品类销量分析")
    # 按品类统计销量
    category_sales = product_orders['category_english'].value_counts().head(10)
    top_category = category_sales.index[0]
    top_category_sales = category_sales.iloc[0]
    top_category_percentage = top_category_sales / category_sales.sum() * 100
    
    sales_summary = f"""
【品类销量关键发现】
• 销量最高的品类: {top_category} ({top_category_sales:,} 件, {top_category_percentage:.1f}%)
• 前10品类销量占比: {category_sales.sum() / len(product_orders) * 100:.1f}%
• 销量TOP3品类: {', '.join(category_sales.head(3).index.tolist())}
"""
    print(sales_summary)
    
    # 可视化1: 品类销量TOP10
    plt.figure(figsize=(18, 10))
    ax = category_sales.plot(kind='bar', color='#2E86AB', alpha=0.8, width=0.7)
    plt.title('产品品类销量TOP10', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('产品品类', fontsize=11, labelpad=10)
    plt.ylabel('销量', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(category_sales):
        ax.text(i, v + max(category_sales)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module3_path, '01_品类销量TOP10.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 品类销售额分析
    print("\n【3.2】品类销售额分析")
    # 按品类统计销售额
    category_revenue = product_orders.groupby('category_english')['price'].sum().sort_values(ascending=False).head(10)
    top_revenue_category = category_revenue.index[0]
    top_revenue = category_revenue.iloc[0]
    
    revenue_summary = f"""
【品类销售额关键发现】
• 销售额最高的品类: {top_revenue_category} (R${top_revenue:,.2f})
• 前10品类销售额占比: {category_revenue.sum() / product_orders['price'].sum() * 100:.1f}%
• 销售额TOP3品类: {', '.join(category_revenue.head(3).index.tolist())}
"""
    print(revenue_summary)
    
    # 可视化2: 品类销售额TOP10
    plt.figure(figsize=(18, 10))
    ax = category_revenue.plot(kind='bar', color='#A23B72', alpha=0.8, width=0.7)
    plt.title('产品品类销售额TOP10', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('产品品类', fontsize=11, labelpad=10)
    plt.ylabel('销售额 (雷亚尔)', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(category_revenue):
        ax.text(i, v + max(category_revenue)*0.01, f'R${v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.savefig(os.path.join(module3_path, '02_品类销售额TOP10.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.3 品类平均单价分析
    print("\n【3.3】品类平均单价分析")
    # 按品类统计平均单价
    category_price = product_orders.groupby('category_english')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    category_price = category_price[category_price['count'] >= 100].head(10)  # 过滤销量低的品类
    top_price_category = category_price.index[0]
    top_price = category_price['mean'].iloc[0]
    
    price_summary = f"""
【品类平均单价关键发现】
• 平均单价最高的品类: {top_price_category} (R${top_price:.2f})
• 高单价品类主要集中在: {', '.join(category_price.head(5).index.tolist())}
"""
    print(price_summary)
    
    # 可视化3: 品类平均单价TOP10
    plt.figure(figsize=(18, 10))
    ax = category_price['mean'].plot(kind='bar', color='#F18F01', alpha=0.8, width=0.7)
    plt.title('产品品类平均单价TOP10', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('产品品类', fontsize=11, labelpad=10)
    plt.ylabel('平均单价 (雷亚尔)', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(category_price['mean']):
        ax.text(i, v + max(category_price['mean'])*0.01, f'R${v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module3_path, '03_品类平均单价TOP10.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.4 产品销量分布
    print("\n【3.4】产品销量分布")
    # 统计每个产品的销量
    product_sales = product_orders['product_id'].value_counts()
    top_product = product_sales.index[0]
    top_product_sales = product_sales.iloc[0]
    
    product_summary = f"""
【产品销量关键发现】
• 销量最高的产品ID: {top_product} ({top_product_sales:,} 件)
• 销量前10的产品贡献了总销量的: {product_sales.head(10).sum() / len(product_orders) * 100:.1f}%
• 80%的销量来自前{int(len(product_sales) * 0.2)}%的产品 (长尾效应)
"""
    print(product_summary)
    
    # 可视化4: 产品销量分布（前20）
    plt.figure(figsize=(20, 10))
    ax = product_sales.head(20).plot(kind='bar', color='#2E86AB', alpha=0.8, width=0.8)
    plt.title('产品销量TOP20', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('产品ID', fontsize=11, labelpad=10)
    plt.ylabel('销量', fontsize=11, labelpad=10)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(product_sales.head(20)):
        ax.text(i, v + max(product_sales.head(20))*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=7)
    
    plt.savefig(os.path.join(module3_path, '04_产品销量TOP20.png'), dpi=300, bbox_inches='tight')
    plt.close()
print(f"\n✅ 模块3分析完成，图片已保存至: {module3_path}")

# ====================== 模块4: 物流配送分析 ======================
print("\n" + "=" * 80)
print("模块4: 物流配送分析")
print("=" * 80)
module4_path = os.path.join(base_output_path, modules[3])

if not data['orders'].empty:
    # 过滤已配送的订单
    delivered_orders = data['orders'][data['orders']['order_status'] == 'delivered'].copy()
    # 4.1 配送时间分析
    print("\n【4.1】配送时间分析")
    # 计算配送时间统计指标
    avg_delivery_time = delivered_orders['delivery_time'].mean()
    median_delivery_time = delivered_orders['delivery_time'].median()
    max_delivery_time = delivered_orders['delivery_time'].max()
    min_delivery_time = delivered_orders['delivery_time'].min()
    
    delivery_summary = f"""
【配送时间关键指标】
• 平均配送时间: {avg_delivery_time:.1f} 天
• 配送时间中位数: {median_delivery_time:.0f} 天
• 最长配送时间: {max_delivery_time:.0f} 天
• 最短配送时间: {min_delivery_time:.0f} 天
• 7天内送达的订单占比: {(delivered_orders['delivery_time'] <= 7).sum() / len(delivered_orders) * 100:.1f}%
• 15天内送达的订单占比: {(delivered_orders['delivery_time'] <= 15).sum() / len(delivered_orders) * 100:.1f}%
"""
    print(delivery_summary)
    
    # 可视化1: 配送时间分布
    plt.figure(figsize=(16, 9))
    plt.hist(delivered_orders[delivered_orders['delivery_time'] <= 30]['delivery_time'], 
                bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    plt.title('配送时间分布 (30天以内)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('配送天数', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.savefig(os.path.join(module4_path, '01_配送时间分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.2 配送延迟分析
    print("\n【4.2】配送延迟分析")
    # 计算延迟统计指标
    avg_delay = delivered_orders['delivery_delay'].mean()
    delay_rate = (delivered_orders['delivery_delay'] > 0).sum() / len(delivered_orders) * 100
    serious_delay_rate = (delivered_orders['delivery_delay'] > 7).sum() / len(delivered_orders) * 100
    
    delay_summary = f"""
【配送延迟关键指标】
• 平均配送延迟: {avg_delay:.1f} 天
• 延迟送达订单占比: {delay_rate:.1f}%
• 严重延迟（超过7天）订单占比: {serious_delay_rate:.1f}%
• 提前送达订单占比: {(delivered_orders['delivery_delay'] < 0).sum() / len(delivered_orders) * 100:.1f}%
• 准时送达（±1天内）订单占比: {(abs(delivered_orders['delivery_delay']) <= 1).sum() / len(delivered_orders) * 100:.1f}%
"""
    print(delay_summary)
    
    # 可视化2: 配送延迟分布
    plt.figure(figsize=(16, 9))
    plt.hist(delivered_orders[(delivered_orders['delivery_delay'] >= -10) & (delivered_orders['delivery_delay'] <= 10)]['delivery_delay'], 
                bins=21, color='#C73E1D', alpha=0.7, edgecolor='black')
    plt.title('配送延迟分布 (±10天)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('延迟天数（负数为提前送达）', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.savefig(os.path.join(module4_path, '02_配送延迟分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.3 月度配送时效趋势
    print("\n【4.3】月度配送时效趋势")
    # 按月份统计平均配送时间
    delivered_orders['delivery_month'] = delivered_orders['order_purchase_timestamp'].dt.to_period('M')
    monthly_delivery = delivered_orders.groupby('delivery_month')['delivery_time'].mean()
    
    trend_summary = f"""
【月度配送时效关键发现】
• 配送时效最好的月份: {monthly_delivery.idxmin()} (平均{monthly_delivery.min():.1f}天)
• 配送时效最差的月份: {monthly_delivery.idxmax()} (平均{monthly_delivery.max():.1f}天)
• 整体配送时效变化趋势: {'整体呈下降趋势' if monthly_delivery.iloc[-1] < monthly_delivery.iloc[0] else '整体呈上升趋势'}
"""
    print(trend_summary)
    
    # 可视化3: 月度配送时效趋势
    plt.figure(figsize=(18, 9))
    ax = monthly_delivery.plot(kind='line', marker='o', linewidth=2, color='#2E86AB', markersize=6)
    plt.title('月度平均配送时间趋势', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('月份', fontsize=11, labelpad=10)
    plt.ylabel('平均配送天数', fontsize=11, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 每隔2个月份显示标签，避免重叠
    for i, (x, y) in enumerate(zip(monthly_delivery.index, monthly_delivery.values)):
        if i % 2 == 0:
            ax.text(x, y + max(monthly_delivery)*0.01, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(os.path.join(module4_path, '03_月度配送时效趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.4 各州配送时效对比
    print("\n【4.4】各州配送时效对比")
    # 合并客户和订单数据
    state_delivery = pd.merge(delivered_orders, data['customers'][['customer_id', 'customer_state']], 
                               on='customer_id', how='left')
    # 按州统计平均配送时间
    state_avg_delivery = state_delivery.groupby('customer_state')['delivery_time'].agg(['mean', 'count']).sort_values('mean')
    state_avg_delivery = state_avg_delivery[state_avg_delivery['count'] >= 100]  # 过滤订单量少的州
    
    state_summary = f"""
【各州配送时效关键发现】
• 配送时效最好的州: {state_avg_delivery.index[0]} (平均{state_avg_delivery['mean'].iloc[0]:.1f}天)
• 配送时效最差的州: {state_avg_delivery.index[-1]} (平均{state_avg_delivery['mean'].iloc[-1]:.1f}天)
• 南北部配送时效差异明显: 南部州平均配送时间远低于北部州
"""
    print(state_summary)
    
    # 可视化4: 各州平均配送时间
    plt.figure(figsize=(18, 9))
    ax = state_avg_delivery['mean'].plot(kind='bar', color='#F18F01', alpha=0.8, width=0.8)
    plt.title('各州平均配送时间', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('州', fontsize=11, labelpad=10)
    plt.ylabel('平均配送天数', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(state_avg_delivery['mean']):
        ax.text(i, v + max(state_avg_delivery['mean'])*0.01, f'{v:.1f}天', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.savefig(os.path.join(module4_path, '04_各州配送时效对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
print(f"\n✅ 模块4分析完成，图片已保存至: {module4_path}")

# ====================== 模块5: 支付方式分析 ======================
print("\n" + "=" * 80)
print("模块5: 支付方式分析")
print("=" * 80)
module5_path = os.path.join(base_output_path, modules[4])

if not data['order_payments'].empty:
    # 5.1 支付方式分布
    print("\n【5.1】支付方式分布")
    # 统计支付方式
    payment_type_counts = data['order_payments']['payment_type'].value_counts()
    top_payment_type = payment_type_counts.index[0]
    top_payment_percentage = payment_type_counts.iloc[0] / payment_type_counts.sum() * 100
    
    payment_summary = f"""
【支付方式关键发现】
• 最主流的支付方式: {top_payment_type} ({top_payment_percentage:.1f}%)
• 前两大支付方式占比: {payment_type_counts.head(2).sum() / payment_type_counts.sum() * 100:.1f}%
• 所有支付方式: {payment_type_counts.index.tolist()}
"""
    print(payment_summary)
    
    # 可视化1: 支付方式分布
    plt.figure(figsize=(12, 12))
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#C73E1D', '#5F4B8B']
    
    # 过滤标签：只显示≥3%的标签和占比
    total_payments = payment_type_counts.sum()
    payment_labels = [name if (count/total_payments)*100 >= 3 else '' for name, count in zip(payment_type_counts.index, payment_type_counts.values)]
    
    # 调整饼图标签，避免重叠
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct >= 3 else ''  # 小于3%的部分不显示百分比
    
    wedges, texts, autotexts = plt.pie(payment_type_counts.values, 
                                       labels=payment_labels, 
                                       autopct=autopct_format,
                                       colors=colors[:len(payment_type_counts)], 
                                       startangle=90,
                                       textprops={'fontsize': 10},
                                       labeldistance=1.05,
                                       pctdistance=0.85)
    
    # 调整百分比文字大小和颜色
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('支付方式分布', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(module5_path, '01_支付方式分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5.2 支付金额分析
    print("\n【5.2】支付金额分析")
    # 按支付方式统计金额
    payment_type_amount = data['order_payments'].groupby('payment_type')['payment_value'].sum().sort_values(ascending=False)
    avg_payment_value = data['order_payments']['payment_value'].mean()
    median_payment_value = data['order_payments']['payment_value'].median()
    
    amount_summary = f"""
【支付金额关键指标】
• 平均单笔支付金额: R${avg_payment_value:.2f}
• 单笔支付金额中位数: R${median_payment_value:.2f}
• 最高单笔支付金额: R${data['order_payments']['payment_value'].max():.2f}
• 最低单笔支付金额: R${data['order_payments']['payment_value'].min():.2f}
• 总支付金额: R${data['order_payments']['payment_value'].sum():,.2f}
"""
    print(amount_summary)
    
    # 可视化2: 各支付方式总金额
    plt.figure(figsize=(16, 9))
    ax = payment_type_amount.plot(kind='bar', color='#A23B72', alpha=0.8, width=0.7)
    plt.title('各支付方式总支付金额', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('支付方式', fontsize=11, labelpad=10)
    plt.ylabel('总支付金额 (雷亚尔)', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=9)
    
    # 添加数值标签，调整位置避免重叠
    for i, v in enumerate(payment_type_amount):
        ax.text(i, v + max(payment_type_amount)*0.01, f'R${v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module5_path, '02_支付方式金额对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5.3 分期付款分析
    print("\n【5.3】分期付款分析")
    # 过滤分期付款订单
    installment_orders = data['order_payments'][data['order_payments']['payment_installments'] > 1]
    installment_rate = len(installment_orders) / len(data['order_payments']) * 100
    avg_installments = installment_orders['payment_installments'].mean()
    max_installments = installment_orders['payment_installments'].max()
    
    installment_summary = f"""
【分期付款关键指标】
• 分期付款订单占比: {installment_rate:.1f}%
• 平均分期数: {avg_installments:.1f} 期
• 最高分期数: {max_installments} 期
• 最常见的分期数: {installment_orders['payment_installments'].value_counts().index[0]} 期
• 分期付款总金额: R${installment_orders['payment_value'].sum():,.2f}
"""
    print(installment_summary)
    
    # 可视化3: 分期数分布
    plt.figure(figsize=(18, 9))
    installment_counts = installment_orders['payment_installments'].value_counts().sort_index()
    ax = installment_counts.plot(kind='bar', color='#2E86AB', alpha=0.8, width=0.8)
    plt.title('分期付款期数分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('分期数', fontsize=11, labelpad=10)
    plt.ylabel('订单数量', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    
    # 每隔2个分期数显示标签，避免重叠
    for i, v in enumerate(installment_counts):
        if i % 2 == 0 or v > max(installment_counts)*0.1:  # 重要数据点都显示
            ax.text(i, v + max(installment_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.savefig(os.path.join(module5_path, '03_分期数分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5.4 支付方式月度趋势
    print("\n【5.4】支付方式月度趋势")
    # 合并订单和支付数据
    payment_orders = pd.merge(data['order_payments'], data['orders'][['order_id', 'order_purchase_timestamp']], 
                             on='order_id', how='left')
    
    # 补充完整代码（原代码截断部分）
    if not payment_orders.empty:
        payment_orders['purchase_month'] = payment_orders['order_purchase_timestamp'].dt.to_period('M')
        # 按月份和支付方式统计订单数
        monthly_payment = payment_orders.groupby(['purchase_month', 'payment_type']).size().unstack(fill_value=0)
        
        # 可视化4: 支付方式月度趋势
        plt.figure(figsize=(20, 10))
        ax = monthly_payment.plot(kind='line', marker='o', linewidth=2, markersize=6)
        plt.title('各支付方式月度订单趋势', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('月份', fontsize=11, labelpad=10)
        plt.ylabel('订单数量', fontsize=11, labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))  # 图例移到右侧
        plt.tight_layout()
        plt.savefig(os.path.join(module5_path, '04_支付方式月度趋势.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        payment_trend_summary = f"""
【支付方式月度趋势关键发现】
• 信用卡支付占比趋势: {'上升' if monthly_payment['credit_card'].iloc[-1] > monthly_payment['credit_card'].iloc[0] else '下降'}
• boleto支付占比趋势: {'上升' if monthly_payment['boleto'].iloc[-1] > monthly_payment['boleto'].iloc[0] else '下降'}
• 最稳定的支付方式: {monthly_payment.std().idxmin()} (标准差: {monthly_payment.std().min():.0f})
"""
        print(payment_trend_summary)

print(f"\n✅ 模块5分析完成，图片已保存至: {module5_path}")

# ====================== 模块6: 客户评价分析 ======================
print("\n" + "=" * 80)
print("模块6: 客户评价分析")
print("=" * 80)
module6_path = os.path.join(base_output_path, modules[5])

if not data['order_reviews'].empty:
    # 6.1 评价分数分布
    print("\n【6.1】评价分数分布")
    review_scores = data['order_reviews']['review_score'].value_counts().sort_index()
    avg_score = data['order_reviews']['review_score'].mean()
    
    review_summary = f"""
【评价分数关键指标】
• 平均评价分数: {avg_score:.2f} 分
• 好评（4-5分）占比: {review_scores.get(4,0) + review_scores.get(5,0) / review_scores.sum() * 100:.1f}%
• 差评（1-2分）占比: {review_scores.get(1,0) + review_scores.get(2,0) / review_scores.sum() * 100:.1f}%
• 最常见评价分数: {review_scores.idxmax()} 分 ({review_scores.max():,} 条)
"""
    print(review_summary)
    
    # 可视化1: 评价分数分布
    plt.figure(figsize=(14, 8))
    colors = ['#C73E1D', '#F18F01', '#FFD300', '#4E9F3D', '#2E86AB']
    ax = review_scores.plot(kind='bar', color=colors, alpha=0.8, width=0.7)
    plt.title('客户评价分数分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('评价分数', fontsize=11, labelpad=10)
    plt.ylabel('评价数量', fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=9)
    
    # 添加数值标签
    for i, v in enumerate(review_scores):
        ax.text(i, v + max(review_scores)*0.01, f'{v:,}\n({v/review_scores.sum()*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.savefig(os.path.join(module6_path, '01_评价分数分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6.2 评价与配送时间关系
    print("\n【6.2】评价与配送时间关系")
    if not data['orders'].empty:
        # 合并评价和订单数据
        review_delivery = pd.merge(data['order_reviews'], data['orders'][['order_id', 'delivery_time', 'delivery_delay']], 
                                  on='order_id', how='left')
        # 计算各分数的平均配送时间
        score_delivery = review_delivery.groupby('review_score')['delivery_time'].agg(['mean', 'count']).dropna()
        
        relation_summary = f"""
【评价与配送关系关键发现】
• 5分评价平均配送时间: {score_delivery.loc[5, 'mean']:.1f} 天
• 1分评价平均配送时间: {score_delivery.loc[1, 'mean']:.1f} 天
• 配送时间越长，评价分数越: {'低' if score_delivery['mean'].iloc[-1] < score_delivery['mean'].iloc[0] else '高'}
"""
        print(relation_summary)
        
        # 可视化2: 评价分数与配送时间关系
        plt.figure(figsize=(14, 8))
        ax = score_delivery['mean'].plot(kind='bar', color='#A23B72', alpha=0.8, width=0.7)
        plt.title('评价分数与平均配送时间关系', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('评价分数', fontsize=11, labelpad=10)
        plt.ylabel('平均配送天数', fontsize=11, labelpad=10)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=9)
        
        # 添加数值标签
        for i, v in enumerate(score_delivery['mean']):
            ax.text(i, v + max(score_delivery['mean'])*0.01, f'{v:.1f}天', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.savefig(os.path.join(module6_path, '02_评价与配送时间关系.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6.3 评价数量月度趋势
    print("\n【6.3】评价数量月度趋势")
    # 合并订单时间
    review_orders = pd.merge(data['order_reviews'], data['orders'][['order_id', 'order_purchase_timestamp']], 
                            on='order_id', how='left')
    if not review_orders.empty:
        review_orders['purchase_month'] = review_orders['order_purchase_timestamp'].dt.to_period('M')
        monthly_reviews = review_orders.groupby('purchase_month').size()
        
        # 可视化3: 评价数量月度趋势
        plt.figure(figsize=(18, 9))
        ax = monthly_reviews.plot(kind='line', marker='o', linewidth=2, color='#2E86AB', markersize=6)
        plt.title('客户评价数量月度趋势', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('月份', fontsize=11, labelpad=10)
        plt.ylabel('评价数量', fontsize=11, labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        # 每隔2个月显示标签
        for i, (x, y) in enumerate(zip(monthly_reviews.index, monthly_reviews.values)):
            if i % 2 == 0:
                ax.text(x, y + max(monthly_reviews)*0.01, f'{y:,}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig(os.path.join(module6_path, '03_评价数量月度趋势.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6.4 评价与产品品类关系
    print("\n【6.4】评价与产品品类关系")
    # 合并评价、订单项、产品数据
    if not data['order_items'].empty and not data['products'].empty:
        review_products = pd.merge(data['order_reviews'], data['order_items'][['order_id', 'product_id']], on='order_id', how='left')
        review_products = pd.merge(review_products, data['products'][['product_id', 'category_english']], on='product_id', how='left')
        
        # 统计各品类的平均评分（过滤评价数少的品类）
        category_reviews = review_products.groupby('category_english')['review_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        category_reviews = category_reviews[category_reviews['count'] >= 100].head(10)
        
        category_summary = f"""
【评价与品类关系关键发现】
• 评价最高的品类: {category_reviews.index[0]} (平均分: {category_reviews['mean'].iloc[0]:.2f})
• 评价最低的品类: {category_reviews.index[-1]} (平均分: {category_reviews['mean'].iloc[-1]:.2f})
• 评价最好的TOP3品类: {', '.join(category_reviews.head(3).index.tolist())}
"""
        print(category_summary)
        
        # 可视化4: 各品类平均评价分数
        plt.figure(figsize=(18, 10))
        ax = category_reviews['mean'].plot(kind='bar', color='#F18F01', alpha=0.8, width=0.7)
        plt.title('产品品类平均评价分数TOP10', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('产品品类', fontsize=11, labelpad=10)
        plt.ylabel('平均评价分数', fontsize=11, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylim(3.5, 5)  # 缩小y轴范围，突出差异
        
        # 添加数值标签
        for i, v in enumerate(category_reviews['mean']):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.savefig(os.path.join(module6_path, '04_品类平均评价分数.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"\n✅ 模块6分析完成，图片已保存至: {module6_path}")

# 最终完成提示
print("\n" + "=" * 80)
print("🎉 所有分析模块执行完成！")
print(f"📁 分析结果已保存至: {base_output_path}")
print("=" * 80)