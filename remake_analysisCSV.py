import pandas as pd
import numpy as np

def donichi_or_syukujitu(df):
    # 土日と祝日の論理和を求めて休日カラムを作成
    # 土日または祝日のいずれかが1の場合、休日=1とする
    df['休日'] = ((df['土日'] == 1) | (df['祝日'] == 1)).astype(int)
    return df


def narasu(df):
    # 数値列を適切な型に変換
    numeric_columns = ['取引ID', '取引月', '曜日', '土日', '祝日', '平均気温', '合計降水量', 
                      '平均風速', '季節', '来客組数', '売上', '数量', '人数', 'isデリバリー',
                      '取引年', '取引月_日時', '取引日_日時', '取引時', '取引分', '取引曜日_日時', '取引日_年間通算',
                      '入店時間_数値', '退店時間_数値', '滞在時間_数値', '取引時刻_数値']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def sincos_henkan(df):
    """
    取引月、曜日、取引日をsin/cos変換する関数
    """
    # 取引日をdatetime型に変換
    df['取引日'] = pd.to_datetime(df['取引日'])
    
    # 取引日のsin/cos変換（年間周期性）
    # 年始からの経過日数を計算（1月1日を1日目とする）
    df['day_of_year'] = df['取引日'].dt.dayofyear
    # 365日で1周期として角度を計算
    angle_day = 2 * np.pi * df['day_of_year'] / 365
    df['取引日_sin'] = np.sin(angle_day)
    df['取引日_cos'] = np.cos(angle_day)
    
    # 取引月のsin/cos変換（12ヶ月周期）
    # 取引月は1～12の値を持つので、12で1周期として角度を計算
    angle_month = 2 * np.pi * (df['取引月'] - 1) / 12  # 取引月を0～11に正規化
    df['取引月_sin'] = np.sin(angle_month)
    df['取引月_cos'] = np.cos(angle_month)
    
    # 曜日のsin/cos変換（7日周期）
    # 曜日は1～7の値を持つので、7で1周期として角度を計算(月をスタート)
    angle_dow = 2 * np.pi * (df['曜日'] - 1) / 7  # 曜日を0～6に正規化
    df['曜日_sin'] = np.sin(angle_dow)
    df['曜日_cos'] = np.cos(angle_dow)
    
    # 中間的な列（day_of_year）は削除
    df = df.drop('day_of_year', axis=1)
    
    return df

def make_dummy(df):
    # 季節をダミー変数化
    # 季節は1=春, 2=夏, 3=秋, 4=冬
    df['季節_春'] = (df['季節'] == 1).astype(int)
    df['季節_夏'] = (df['季節'] == 2).astype(int)
    df['季節_秋'] = (df['季節'] == 3).astype(int)
    df['季節_冬'] = (df['季節'] == 4).astype(int)

    """
    部門名をダミー変数化
    """
    if '部門名' in df.columns:
        # 部門名の種類を確認
        unique_bumon = df['部門名'].dropna().unique()
        print(f"部門名の種類: {list(unique_bumon)} (計{len(unique_bumon)}種類)")
        
        # 各部門名に対してダミー変数を作成
        for bumon in unique_bumon:
            # 部門名を安全なカラム名に変換（特殊文字を除去）
            safe_bumon_name = str(bumon).replace(' ', '_').replace('/', '_').replace('-', '_')
            column_name = f'部門_{safe_bumon_name}'
            df[column_name] = (df['部門名'] == bumon).astype(int)
            
        print(f"部門名のダミー変数化完了: {len(unique_bumon)}個の列を追加")
        
        # ダミー変数の統計を表示
        for bumon in unique_bumon:
            safe_bumon_name = str(bumon).replace(' ', '_').replace('/', '_').replace('-', '_')
            column_name = f'部門_{safe_bumon_name}'
            count = df[column_name].sum()
            percentage = count / len(df) * 100
            print(f"  {bumon}: {count}件 ({percentage:.1f}%)")
            
    else:
        print("部門名列が見つかりません")
    
    return df

def jikan_henkan(df):
    """
    滞在時間、入店時間、退店時間、取引日時を数値データに変換する関数
    """
    print("\n--- 時間データの数値変換開始 ---")
    
    # 滞在時間の処理（"0 days HH:MM:SS"形式を時間単位の数値に変換）
    if '滞在時間' in df.columns:
        # タイムデルタ形式に変換
        df['滞在時間_timedelta'] = pd.to_timedelta(df['滞在時間'], errors='coerce')
        
        # 時間単位の数値に変換（分を60で割る）
        df['滞在時間_数値'] = df['滞在時間_timedelta'].dt.total_seconds() / 3600  # 時間単位
        
        # 統計情報を表示
        valid_stay = df['滞在時間_数値'].dropna()
        if len(valid_stay) > 0:
            print(f"滞在時間変換完了: {len(valid_stay)}件")
            print(f"  平均: {valid_stay.mean():.2f}時間")
            print(f"  最小値: {valid_stay.min():.2f}時間")
            print(f"  最大値: {valid_stay.max():.2f}時間")
            print(f"  例: '0 days 01:00:00' → {1.0}, '0 days 00:30:00' → {0.5}")
        
        # 中間列を削除
        df = df.drop('滞在時間_timedelta', axis=1)
    
    print("時間データの数値変換完了\n")
    return df

def delivery_henkan(df):
    """
    人数列の「デリバリー」を基準にisデリバリー属性を作成し、人数を修正する関数
    """
    print("\n--- デリバリーデータの処理開始 ---")
    
    if '人数' in df.columns:
        # 元の人数列の状況を確認
        unique_values = df['人数'].value_counts()
        print(f"人数列の値の分布:")
        for value, count in unique_values.items():
            print(f"  '{value}': {count}件")
        
        # デリバリーフラグを作成
        df['isデリバリー'] = (df['人数'] == 'デリバリー').astype(int)
        
        # デリバリーの件数を表示
        delivery_count = df['isデリバリー'].sum()
        total_count = len(df)
        delivery_ratio = delivery_count / total_count * 100
        
        print(f"\nデリバリー判定結果:")
        print(f"  デリバリー: {delivery_count}件 ({delivery_ratio:.1f}%)")
        print(f"  店内利用: {total_count - delivery_count}件 ({100 - delivery_ratio:.1f}%)")
        
        # デリバリー以外のデータから人数の平均を計算
        non_delivery_data = df[df['人数'] != 'デリバリー']['人数']
        # 数値に変換できるもののみを対象
        numeric_ninzu = pd.to_numeric(non_delivery_data, errors='coerce').dropna()
        
        if len(numeric_ninzu) > 0:
            avg_ninzu = numeric_ninzu.mean()
            print(f"\n店内利用の平均人数: {avg_ninzu:.2f}人")
            
            # 人数列の「デリバリー」を平均値に修正
            df.loc[df['人数'] == 'デリバリー', '人数'] = str(round(avg_ninzu, 1))
            
            print(f"デリバリーの人数を平均値 {round(avg_ninzu, 1)} で補完しました")
        else:
            print("平均値を計算できませんでした。0で補完します。")
            # 人数列の「デリバリー」を0に修正
            df.loc[df['人数'] == 'デリバリー', '人数'] = '0'
        
        # 修正後の人数列の分布を確認
        print(f"\n修正後の人数列の分布:")
        unique_values_after = df['人数'].value_counts().sort_index()
        for value, count in unique_values_after.items():
            print(f"  {value}人: {count}件")
        
        print("デリバリーデータの処理完了\n")
    else:
        print("人数列が見つかりません\n")
    
    return df

def main():
    print("Hello from demo-proj!")

    df = pd.read_csv(  # 読み込み
        "./csv/data_analysis.csv",  # 入力ファイル
        dtype=str  # 値が化けないように
    ) 
    
    print(f"元データ: {len(df)}件、{len(df.columns)}列")
    
    df = delivery_henkan(df)  # デリバリーデータの処理
    df = narasu(df)  # 数値列を数値型に変換
    df = jikan_henkan(df)  # 時間データを数値に変換
    df = make_dummy(df)  # ダミー変数化（部門、季節）
    df = sincos_henkan(df)  # 取引月、曜日、取引日をsin/cos変換
    df = donichi_or_syukujitu(df)  # 土日祝日を休日に変換
    
    df.to_csv(  # 書き出し
        "./out_csv/demo_analysis.csv",  # 出力ファイル
        index=False,  # read_csv()で付加されたindexを除去
        encoding='utf-8-sig'  # 日本語対応
    )
    print(f"\n最終データを保存しました: ./out_csv/demo_analysis.csv")
    print(f"最終データ: {len(df)}件、{len(df.columns)}列")
    
    # 追加された列の概要を表示
    bumon_columns = [col for col in df.columns if col.startswith('部門_')]
    if bumon_columns:
        print(f"\n--- 追加された部門ダミー変数 ---")
        for col in bumon_columns:
            print(f"  {col}")
    
    kisetsu_columns = [col for col in df.columns if col.startswith('季節_')]
    if kisetsu_columns:
        print(f"\n--- 季節ダミー変数 ---")
        for col in kisetsu_columns:
            print(f"  {col}")
    
    # 時間関連の列の概要を表示
    jikan_columns = [col for col in df.columns if any(x in col for x in ['時間_数値', '時刻_数値', '滞在時間_数値'])]
    if jikan_columns:
        print(f"\n--- 追加された時間関連変数 ---")
        for col in jikan_columns:
            print(f"  {col}")
            
    # 滞在時間の統計情報
    if '滞在時間_数値' in df.columns:
        stay_stats = df['滞在時間_数値'].describe()
        print(f"\n--- 滞在時間の統計（時間単位） ---")
        print(f"  件数: {stay_stats['count']:.0f}件")
        print(f"  平均: {stay_stats['mean']:.2f}時間")
        print(f"  中央値: {stay_stats['50%']:.2f}時間")
        print(f"  最小値: {stay_stats['min']:.2f}時間")
        print(f"  最大値: {stay_stats['max']:.2f}時間")
    
    # デリバリー関連の統計情報
    if 'isデリバリー' in df.columns:
        delivery_stats = df['isデリバリー'].value_counts()
        print(f"\n--- デリバリー統計 ---")
        print(f"  店内利用: {delivery_stats.get(0, 0)}件")
        print(f"  デリバリー: {delivery_stats.get(1, 0)}件")
        
        # 人数別の統計（デリバリー除く）
        if '人数' in df.columns:
            ninzu_stats = df[df['isデリバリー'] == 0]['人数'].value_counts().sort_index()
            print(f"\n--- 人数分布（店内利用のみ） ---")
            for ninzu, count in ninzu_stats.items():
                print(f"  {ninzu}人: {count}件")

if __name__ == "__main__":
    main()
