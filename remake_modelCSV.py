import pandas as pd
import numpy as np

def narasu(df):
    # 数値列を適切な型に変換
    numeric_columns = ['取引月', '曜日', '土日', '祝日', '平均気温', '合計降水量', 
                      '平均風速', '季節', '来客組数', '売上']
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
    return df


def donichi_or_syukujitu(df):
    # 土日と祝日の論理和を求めて休日カラムを作成
    # 土日または祝日のいずれかが1の場合、休日=1とする
    df['休日'] = ((df['土日'] == 1) | (df['祝日'] == 1)).astype(int)
    return df

# 平日.csvと休日.csvに分割して保存
def split_kyuujitu_heijitu(df):
    heijitsu_df = df[df['休日'] == 0].copy()  # 休日=0（平日）
    kyujitsu_df = df[df['休日'] == 1].copy()  # 休日=1（土日祝日）

    # 平日データを保存
    heijitsu_df.to_csv(
        "./out_csv/平日.csv",
        index=False,
        encoding='utf-8-sig'  # 日本語ファイル名対応
    )
    print(f"平日データを保存しました: ./out_csv/平日.csv ({len(heijitsu_df)}件)")
    # 休日データを保存
    kyujitsu_df.to_csv(
        "./out_csv/休日.csv",
        index=False,
        encoding='utf-8-sig'  # 日本語ファイル名対応
    )
    print(f"休日データを保存しました: ./out_csv/休日.csv ({len(kyujitsu_df)}件)")

def main():
    print("Hello from remake_csv!")
    df = pd.read_csv(  # 読み込み
        "./csv/model_data.csv",  # 入力ファイル
        dtype=str  # 値が化けないように
    )

    df = narasu(df)  # 数値列を数値型に変換

    df = donichi_or_syukujitu(df)  # 土日祝日を休日に変換
    
    df = sincos_henkan(df)  # 取引月、曜日、取引日をsin/cos変換

    df = make_dummy(df)  # 季節をダミー変数化

    # 全体のデータを保存
    df.to_csv(  # 書き出し
        "./out_csv/demo_data.csv",  # 出力ファイル
        index=False  # read_csv()で付加されたindexを除去
    )
    print(f"全データを保存しました: ./out_csv/demo_data.csv ({len(df)}件)")
    
    # 平日と休日に分割して保存
    split_kyuujitu_heijitu(df)

if __name__ == "__main__":
    main()
