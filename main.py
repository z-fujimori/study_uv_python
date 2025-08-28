import pandas as pd
import numpy as np
from import_edit_file import edit, analyze_data, save_analysis_results  # 分析関数も追加でimport

def main():
    print("Hello from demo-proj!")

    df = pd.read_csv(  # 読み込み
        "./csv/model_data.csv",  # 入力ファイル
        dtype=str  # 値が化けないように
    )

    df = edit(df)  # データ加工

    df.to_csv(  # 書き出し
        "./csv/output_model.csv",  # 出力ファイル
        index=False  # read_csv()で付加されたindexを除去
    )
    
    print("前処理済みデータを保存しました: ./csv/output_model.csv")
    
    # データ分析を実行
    results = analyze_data(df)
    
    # 分析結果を保存
    save_analysis_results(results)
    
    print("\n=== 分析完了 ===")
    print("特徴量重要度と予測結果がcsvフォルダに保存されました。")





if __name__ == "__main__":
    main()
