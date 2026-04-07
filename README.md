# LLM マルチエージェント間の相互作用の分析
**Analysis of Interactions between LLM Multi-Agents**

> [NLP2025](https://www.anlp.jp/nlp2025/)（言語処理学会第31回年次大会）にて発表（2025年3月）

---

## 概要

本プロジェクトでは、社会心理学の視点から複数のLLMエージェント間の相互作用を分析します。社会心理学においてグループの問題解決パフォーマンスを説明する理論である **Steiner's theory** を、LLMマルチエージェント（MA）システムに適用し、エージェント間の相互作用がグループ全体のパフォーマンスにどのような影響を与えるかを定量的に分析しました。

### 主な発見

- **誤選択の評価値と正誤は負の相関** --- 全条件において確認され、Steiner's theoryと合致
- **誤生成の評価値と正誤には大きな負の相関** --- 誤った情報の伝搬がMAに悪影響を及ぼすことを示唆
- 修正や新案といった相互作用の有効性は **タスクやフレームワークの種類に依存**

## 理論的背景

**Steiner's Theory (1972):**

```
Actual Group Productivity = Potential Productivity
                          - Productivity Losses
                          + Productivity Gains
```

Steiner's theoryのProductivity LossesとProductivity Gainsの中から、LLMで定量的に評価可能な項目を選定し、以下の評価指標を定義しました。

| カテゴリ | 指標 | 説明 |
|---|---|---|
| **Productivity Losses** | 誤選択 | 間違ったアイデアを別のエージェントが選択する数 |
| | 誤生成 | エージェントが間違った回答を生み出した数 |
| | 同意見 | 前のエージェントと同じアイデアが繰り返される数 |
| **Productivity Gains** | 修正 | あるアイデアが別のエージェントによって訂正された数 |
| | 新案 | シングルエージェントの回答にはない、新しい回答の数 |

## 実験設定

### データセット
- **BBH（Causal Judgment）**: 200問 --- 因果関係を推論する難易度の高い推論タスク
- **GSM8K（Mathematical Reasoning）**: 全187問 --- 小学校レベルの数学の問題

### MAフレームワーク
- **Cooperative**（Du et al., 2023）: 2つのエージェントが協力し、回答を共有しながら課題を解く
- **Competitive**（Liang et al., 2024）: 2人の参加者が対立する視点から議論し、ジャッジが評価

### モデル
- **GPT-4o**（OpenAI）を全エージェントのバックボーンLLMとして使用

## 結果

### 各条件における正答率

| 条件 | 正答率 (%) |
|---|---|
| GSM8K-SA-4o | 93.0 |
| GSM8K-Cooperative-4o | 97.5 |
| GSM8K-Competitive-4o | 94.5 |
| BBH-SA-4o | 69.5 |
| BBH-Cooperative-4o | 65.2 |
| BBH-Competitive-4o | 55.1 |

容易なタスク（GSM8K）ではMAが正答率を向上させる一方、困難なタスク（BBH）ではむしろ性能が低下する結果となりました。エージェント間の相互作用の分析が重要であることを示しています。

## プロジェクト構成

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── src/
│   ├── llm_utils.py                 # LLM APIラッパー（GPT-4o / GPT-4o-mini）
│   ├── cooperative_framework.py     # Cooperative MAフレームワーク（Streamlit UI）
│   ├── competitive_framework.py     # Competitive MA（議論）フレームワーク（Streamlit UI）
│   ├── single_agent.py              # シングルエージェント（SA）ベースライン
│   ├── batch_runner.py              # バッチ実験実行スクリプト
│   ├── evaluate_metrics.py          # Steiner's theoryに基づく指標評価（GPT-4o使用）
│   └── analyze_results.py           # 相関分析・正答率算出
├── results/
│   ├── bbh_cooperative/             # BBH x Cooperative 結果（30問）
│   ├── bbh_competitive/             # BBH x Competitive 結果（30問）
│   ├── gsm8k_cooperative/           # GSM8K x Cooperative 結果（30問）
│   └── gsm8k_competitive/           # GSM8K x Competitive 結果（30問）
├── examples/
│   ├── cooperative_*.md             # Cooperativeの会話ログ例
│   └── competitive_*.md             # Competitiveの会話ログ例
└── docs/
    ├── NLP2025_paper.pdf            # 論文
    └── NLP2025_slides.pdf           # 発表スライド
```

## セットアップ

### 前提条件
- Python 3.9+
- OpenAI APIキー（GPT-4oアクセス権）

### インストール

```bash
git clone https://github.com/SodaShikenn/LLM-Multi-Agent-Interaction-Analysis.git
cd LLM-Multi-Agent-Interaction-Analysis
pip install -r requirements.txt
```

### 環境設定

```bash
cp .env.example .env
# .env を編集し、OpenAI APIキーを設定
```

### インタラクティブデモの実行

```bash
cd src

# Cooperativeフレームワーク
streamlit run cooperative_framework.py

# Competitiveフレームワーク（議論）
streamlit run competitive_framework.py
```

### 実験パイプラインの実行

```bash
cd src

# Step 1: シングルエージェントベースラインの実行
python single_agent.py --dataset ../data/bbh.json --name bbh

# Step 2: マルチエージェント実験の実行
python batch_runner.py --dataset ../data/bbh.json --name bbh --framework cooperative
python batch_runner.py --dataset ../data/bbh.json --name bbh --framework competitive
python batch_runner.py --dataset ../data/gsm8k.json --name gsm8k --framework cooperative
python batch_runner.py --dataset ../data/gsm8k.json --name gsm8k --framework competitive

# Step 3: 指標評価（Steiner's theory）
python evaluate_metrics.py \
    --ma_dir ../results/bbh_cooperative \
    --sa_dir ../results/bbh_single_agent \
    --dataset ../data/bbh.json \
    --output ../results/evaluation/bbh_cooperative_metrics.json

# Step 4: 相関分析（論文の表3〜6を再現）
python analyze_results.py --all_dir ../results/evaluation/
```

## 技術スタック

- **LLM**: GPT-4o (OpenAI)
- **オーケストレーション**: LangGraph + LangChain
- **UI**: Streamlit
- **分析**: NumPy（相関係数行列）
- **理論基盤**: Steiner's Group Productivity Theory（社会心理学）

## 著者

東京大学, 2025

## 参考文献

1. Du, Y. et al. "Improving factuality and reasoning in language models through multiagent debate." (2023)
2. Liang, T. et al. "Encouraging divergent thinking in large language models through multi-agent debate." (2024)
3. Steiner, I.D. "Group process and productivity." (1972)
4. Hill, G.W. "Group versus individual performance: Are n+1 heads better than one?" Psychological Bulletin. (1982)
5. Zhang, J. et al. "Exploring collaboration mechanisms for LLM agents: A social psychology view." ACL 2024.

## ライセンス

[MIT License](LICENSE)
