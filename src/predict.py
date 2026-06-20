import os
import torch
import torch.nn.functional as F
import joblib
import jieba
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import config
from src import data_loader
from src import models

class MultiModelPredictor:
    def __init__(self):
        print("[加载模型] 正在初始化多模型预测器，加载所有训练好的权重...")
        
        # 1. 加载 Baseline
        self.baseline_model = None
        if os.path.exists(config.BASELINE_WEIGHT_PATH):
            self.baseline_data = joblib.load(config.BASELINE_WEIGHT_PATH)
            self.vectorizer = self.baseline_data["vectorizer"]
            self.lr_model = self.baseline_data["model"]
            self.baseline_model = True
            print(" - TF-IDF + Logistic Regression 加载成功。")
        else:
            print(" - [警告] TF-IDF + Logistic Regression 权重文件不存在。")

        # 2. 加载 BiLSTM + Attention
        self.lstm_model = None
        vocab_path = os.path.join(config.OUTPUT_DIR, "bilstm_vocab.json")
        if os.path.exists(config.LSTM_WEIGHT_PATH) and os.path.exists(vocab_path):
            self.vocab = data_loader.Vocab.load(vocab_path)
            self.lstm_model = models.BiLSTMAttention(
                vocab_size=len(self.vocab.word2idx),
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT
            ).to(config.DEVICE)
            self.lstm_model.load_state_dict(torch.load(config.LSTM_WEIGHT_PATH, map_location=config.DEVICE))
            self.lstm_model.eval()
            print(" - BiLSTM + Attention 加载成功。")
        else:
            print(" - [警告] BiLSTM + Attention 权重或词表不存在。")

        # 3. 加载 BERT (RoBERTa)
        self.bert_model = None
        if os.path.exists(config.BERT_WEIGHT_PATH):
            self.bert_tokenizer = AutoTokenizer.from_pretrained(config.BERT_WEIGHT_PATH)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(config.BERT_WEIGHT_PATH).to(config.DEVICE)
            self.bert_model.eval()
            print(" - BERT (RoBERTa) 模型与分词器加载成功。")
        else:
            print(" - [警告] BERT 权重模型目录不存在。")

    def predict(self, text):
        """
        对输入的文本进行风格识别预测
        """
        text = text.strip()
        if not text:
            return
            
        print("\n" + "-"*30 + f" 预测输入: \"{text}\" " + "-"*30)
        
        # 1. 逻辑回归预测
        if self.baseline_model:
            X_vec = self.vectorizer.transform([text])
            probs = self.lr_model.predict_proba(X_vec)[0]
            # label 1 为网络梗风格
            prob_meme = probs[1]
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[TF-IDF + LR]  网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")
        
        # 2. BiLSTM + Attention 预测与注意力权重分析
        if self.lstm_model:
            words = list(jieba.cut(text))
            indices = self.vocab.text_to_indices(text, config.LSTM_MAX_LEN)
            inputs = torch.tensor([indices], dtype=torch.long).to(config.DEVICE)
            
            with torch.no_grad():
                logits, attn_weights = self.lstm_model(inputs)
                probs = F.softmax(logits, dim=1)[0]
                prob_meme = probs[1].item()
                
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[BiLSTM+Attn]  网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")
            
            # 可视化注意力机制所关注的词
            # 取出当前文本长度对应的注意力权重
            attn_scores = attn_weights[0][:len(words)].cpu().numpy()
            if len(attn_scores) > 0:
                # 归一化显示
                attn_scores = attn_scores / (np.sum(attn_scores) + 1e-9)
                print(" -> 词汇注意力权重分析:")
                attn_str_list = []
                for word, score in zip(words, attn_scores):
                    attn_str_list.append(f"{word}({score*100:.1f}%)")
                print("    " + " | ".join(attn_str_list))
                
        # 3. BERT 预测
        if self.bert_model:
            encoding = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=config.BERT_MAX_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(config.DEVICE)
            attention_mask = encoding["attention_mask"].to(config.DEVICE)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)[0]
                prob_meme = probs[1].item()
                
            style_name = "网络梗/抽象语体" if prob_meme >= 0.5 else "日常/标准语体"
            print(f"[BERT (RoB)]   网络梗概率: {prob_meme*100:.2f}% -> 风格判定: {style_name}")

def interactive_loop():
    import random
    predictor = MultiModelPredictor()
    print("\n" + "="*25 + " 中文互联网语言风格识别测试 " + "="*25)
    print("输入 'exit' 或 'q' 退出程序。")
    print("现在可以输入您希望测试的句子，系统将实时输出三种模型的预测结果：\n")
    
    # 至少100句中文互联网热梗、神人梗、地狱梗与地狱笑话候选池
    memes_pool = [
        "大型纪录片《麦克阿瑟传奇》为您播出：如果遇到他，我建议直接投降。",
        "上帝问我想要什么，我说想要世界和平，上帝笑了；我说想要你期末不挂科，上帝沉默了，并开始帮我找工作。",
        "肯德基疯狂星期四，V我50，听我讲如何用BERT识别你的抽象指数。",
        "牢大，别坐直升机了，我想你了！",
        "你敲电子木鱼，佛祖在天上用功德加特林超度你。",
        "今天的地狱笑话很赞，阎王已经在生死簿上给我留了个好位置。",
        "伤害性不大，侮辱性极强，小丑竟是我自己！",
        "你这人有点意思，但不多，建议回炉重造。",
        "泰裤辣！这波操作直接在大气层，我愿称你为最强。",
        "电子功德-100，地狱VIP通道已为您自动开启。",
        "建议佛祖连夜把灵山的大理石地板换成防滑垫。",
        "上帝关上了一扇门，却忘记给你留个窗，顺便把空调也关了。",
        "你这算盘珠子都崩我脸上了，老铁。",
        "电子木鱼：你再敲，我都要变成赛博舍利子了。",
        "小丑竟在我身边，原来我自己就是最大的马戏团。",
        "尊嘟假嘟？我觉得你是在无中生有，暗度陈仓。",
        "这波属于是神仙打架，我直接一个好家伙。",
        "大肠里故意留了一点，是怕你吃不出原汁原味。",
        "马保国：年轻人不讲武德，来骗，来偷袭，我69岁的老同志。",
        "接化发！闪电五连鞭，直接把你整不会了。",
        "姬霓太美，你实在是太baby了。",
        "我直接一个大无语，这操作秀得我头皮发麻。",
        "如果地狱有段位，你现在起码是个最强王者。",
        "地府判官看了一眼你的功德，连夜把生死簿改成了EXCEL表格。",
        "我佛慈悲，但不渡铁憨憨。",
        "阎王：你先别死，我这生死簿的服务器装不下你的罪孽。",
        "真·赛博超度：用Python脚本自动敲电子木鱼增加功德。",
        "上帝说要有光，于是你拉开了裤子拉链。",
        "虽然你没有功德，但你成功逗笑了地狱的恶魔。",
        "地府第一条规矩：不要在判官面前讲地狱笑话。",
        "你这功德扣得，连孟婆汤都要兑水了。",
        "当代大学生三大幻觉：我能及格、她喜欢我、今天不降温。",
        "奥利给！干了这碗恒河水，来生还做印度人。",
        "建议直接买张站票，连夜买站票逃跑。",
        "我的天呐，这简直是针不戳，住在山里针不戳。",
        "只要我没有道德，你就没办法绑架我。",
        "摆烂也是一种态度，我直接原地躺平。",
        "虽然我很穷，但我有一颗想要V我50的心。",
        "麦克阿瑟：如果我在战场上遇到写这个代码的人，我会选择当场退伍。",
        "上帝创造了人类，人类创造了地狱笑话，上帝看后直呼内行。",
        "你的功德箱空空如也，甚至还欠了地府高利贷。",
        "阎王爷看你的简历，都得先深吸一口凉气。",
        "佛祖：我劝你善良，不然我一巴掌把你拍成赛博舍利。",
        "别敲了别敲了，木鱼都敲出火花塞了！",
        "判官：你这小伙子有前途，地府正好缺个写Python的。",
        "你在地狱的床位已经升到了总统套房。",
        "地狱笑话确实好笑，就是有点废功德。",
        "功德+1，但因为你笑了，功德-100。",
        "电子菩萨在线超度，只需9.9元功德包邮到家。",
        "地府VIP通道专属客服正在为您接入。",
        "这波直接送你上西天，佛祖接引，免排队。",
        "上帝：你别进天堂了，我怕你把天使带坏了。",
        "你这功德扣的，直接把阎王爷的CPU干烧了。",
        "别问，问就是尊嘟假嘟，问就是泰裤辣。",
        "小丑表演结束，谢谢大家，可以散场了。",
        "大型纪录片《关于我敲木鱼却下地狱这件事》正在热播。",
        "佛祖听完你的祈祷，默默把金身换成了防弹衣。",
        "判官：你这是在功德簿上蹦迪啊。",
        "阎王都得给你递根烟，顺便问你地狱笑话的后续。",
        "这波操作秀得我满地找头。",
        "上帝：我这里是天堂，不是地狱笑话分享大会。",
        "你这功德值，连十八层地狱都只能算地下室。",
        "电子木鱼自动敲击器：一秒百敲，功德爆表。",
        "佛祖：你看我像是不想要你功德的样子吗？",
        "判官翻了翻生死簿，发现你的功德栏写着‘暂无额度’。",
        "你这地狱笑话，直接把孟婆的汤都给整咸了。",
        "上帝给你的脑子注了水，但忘记给你安排水管了。",
        "地府的油锅已经为你预热到了180度，金黄酥脆。",
        "当代打工人现状：钱没赚到，功德扣了不少。",
        "只要跑得够快，阎王的生死簿就追不上我。",
        "你这功德扣得，连地狱犬看你都直摇头。",
        "佛祖：我虽然普渡众生，但你这种我得先打个申请报告。",
        "判官：你的功德余额已不足，请及时充值。",
        "你这笑话，直接让地府的温度下降了十度。",
        "上帝：我原谅你的罪恶，但地狱不原谅。",
        "阎王：你这情况，得直接送去重铸。",
        "敲木鱼只能加功德，但你这行为得扣高利贷。",
        "佛祖：我的金身是用来防小人的，不是防你的地狱笑话的。",
        "判官连夜加班，就是为了给你整理罪行清单。",
        "你这人，阎王见了都得先查查后台数据。",
        "地狱的恶魔都开始学你的地狱笑话了。",
        "上帝关门的时候，顺便把你的智商也给夹了。",
        "你这功德，直接扣到了负无穷大。",
        "佛祖：我佛不渡铁头娃，更不渡嘴硬的。",
        "判官：你的生死簿已转为PDF格式，不可修改。",
        "阎王：我这地府，迟早被你这笑话给冻住。",
        "电子木鱼已损坏，请联系赛博菩萨进行售后维修。",
        "你这功德扣得，连牛头马面都要给你众筹了。",
        "上帝：你这孩子，天堂有路你不走，地狱无门你自创地狱笑话。",
        "佛祖：你看我像不像是要超度你的样子？",
        "判官默默掏出了计算器，开始算你欠了多少功德。",
        "阎王：你这简历，连地狱编制都进不去。",
        "这地狱笑话，直接把奈何桥给整塌了。",
        "你的功德箱已经被你敲成了骨灰盒。",
        "佛祖：我今天放假，有事请找赛博观音。",
        "判官：你的名字已经在生死簿的首尾两端都置顶了。",
        "阎王：你再讲地狱笑话，我就把你送去天堂折磨上帝。",
        "上帝：求求你别来天堂，我怕这里的圣光亮瞎你的狗眼。",
        "电子木鱼敲击中：功德+0.01，地狱门票打折中。",
        "你这功德扣的，直接引起了阴曹地府的通货膨胀。",
        "佛祖听了你的笑话，连夜搬家到了别的星系。",
        "判官：你的罪孽已经可以用大语言模型来做语义分析了。",
        "阎王：别敲木鱼了，我地府的KPI都快被你扣超标了。",
        "这波啊，这波是阎王敲门——胆小鬼在作死。",
        "上帝拍了拍你的脑瓜子，发现里面有一句疯狂星期四V我50。"
    ]

    while True:
        try:
            user_input = input("\n请输入测试文本 >>> ")
            if user_input.strip().lower() in ['exit', 'q']:
                print("\n[退出测试]")
                random_meme = random.choice(memes_pool)
                print(f"🎁 随机网络热梗赠言: \"{random_meme}\"\n")
                break
            predictor.predict(user_input)
        except KeyboardInterrupt:
            print("\n[退出测试]")
            random_meme = random.choice(memes_pool)
            print(f"🎁 随机网络热梗赠言: \"{random_meme}\"\n")
            break
        except Exception as e:
            print(f"预测出错: {e}")

if __name__ == "__main__":
    interactive_loop()
