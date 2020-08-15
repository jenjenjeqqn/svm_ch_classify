# -*- coding: UTF-8 -*-
import uvicorn as u
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import jieba
import os
from sklearn.neural_network import MLPClassifier
import sklearn.svm
from sklearn.multiclass import OneVsRestClassifier
import joblib


app = FastAPI()

class Mod(BaseModel):
    doc: str


# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("logs/log" + time.strftime("%Y%m%d", time.localtime()) + ".txt", encoding='gbk')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
# 结束logger配置


good_words = [{"cate": "新冠", "word_list": ['新型冠状病毒', '新冠病毒', '冠状病毒', '新冠肺炎', '新冠疫苗', '抗疫', '疫情']},
            {"cate": "军事",
             "word_list": ["军事", "宇航局", "航天", "航空", "海军", "空军", "国防", "领土", "战争", "军队", "作战", "部队", "武器", "士兵"]},
            {"cate": "创新", "word_list": ['创新', '知识产权', '专利', '创业', '新兴', '孵化', '双创']},
            {"cate": "科技", "word_list": ['科技', '科学技术', '谷歌', '微软', '人工智能', '芯片', '大数据', '机器学习']},
            {"cate": "政治", "word_list": ['政策', '政权', '政治', '国家发展', '政府', '制度', '国务院', '监管']},
            {"cate": "社会", "word_list": ['社会', '司法', '教育', '就业', '生活', '百姓', '治理', '建设']},
            {"cate": "经济", "word_list": ['经济', '财政', 'GDP', '消费', '金融', '货币', '市场', '投资', '银行']},
            {"cate": "外交",
             "word_list": ['外交', '友好合作', '伙伴关系', '国际形势', '国际关系', '大使馆', '全球化', '国际社会', '金砖', '峰会', '会晤', '涉外']}
            ]

for i in good_words:
    for j in i["word_list"]:
        jieba.add_word(j)


def preprocess(stopword_path, data_Path):
    # 加载停用词
    fp = open(stopword_path, encoding='UTF-8')
    stop_words = fp.readlines()
    fp.close()
    sw = []
    for i in stop_words:
        sw.append(i.replace('\n', ''))

    # 读取目录
    dirList = os.listdir(data_Path)
    filedirs = []
    for i in dirList:
        filedirs.append(str(i) + '/')

    # 中文文本需要分词，构成语料库
    for dir in filedirs:
        if os.path.exists('./corpus/'+dir):
            print('yes, folder exists')
            pass
        else:
            os.mkdir('./corpus/'+dir)
        logger.info('正在分词，写入文件夹 ./corpus/%s' % dir)
        for file in os.listdir(data_Path+'/'+dir):
            f = open(data_Path+'/'+dir+file, encoding='utf8')
            f1 = open('./corpus/'+dir+file, 'w', encoding='utf8')
            text = f.read().strip()
            words = jieba.cut(text)
            doc = ''
            for word in words:
                if word not in sw:
                    doc = doc + " " + word
            f1.write(doc)
            f.close()
            f1.close()


def train_model(newsCH, save_path1, save_path2):
    logger.info('加载外部数据集')
    print(newsCH.target_names)
    logger.info('划分训练集、测试集')
    x_train, x_test, y_train, y_test = train_test_split(newsCH.data, newsCH.target, random_state=1)

    logger.info('词向量化')
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(x_train)
    print(vectors.shape)
    joblib.dump(vectorizer, save_path1)
    # 测试集使用 transform； 注意fit_transform 和 transform 的区别
    vectors_test = vectorizer.transform(x_test)

    logger.info('训练svc 多分类模型')
    svc = OneVsRestClassifier(sklearn.svm.SVC(kernel='linear',probability=True))
    svc.fit(vectors, y_train)
    print('SVC Testing Score: %.2f' % svc.score(vectors_test, y_test))
    print(svc.predict(vectors_test))
    joblib.dump(svc, save_path2)

    # 划分训练集、测试集
    # # MultinomialNB 训练模型
    # clf = MultinomialNB(alpha=0.1)
    # clf.fit(x_train, y_train)
    # print('Training Score: %.2f' % clf.score(x_train, y_train))
    # print('Multinomial NB Testing Score: %.2f' % clf.score(x_test, y_test))
    # print(clf.predict(x_test))
    # joblib.dump(clf, '''nb_train_model.m''')
    # #
    # # # 神经网络 MLP 训练模型
    # # mlp = MLPClassifier(random_state=7, max_iter=20).fit(x_train, y_train)
    # # print('神经网络 MLP Testing Score: %.2f' % mlp.score(x_test, y_test))
    # # print(mlp.predict(x_test))
    # # joblib.dump(mlp, '''mlp_train_model.m''')


def svc_predict(newsCH, model_path1, model_path2, doc):
    logger.info('调用模型')
    vectorizer = joblib.load(model_path1)
    svc = joblib.load(model_path2)

    words = jieba.cut(doc)
    # 加载停用词
    fp = open('stopwords_cn.txt', encoding='UTF-8')
    stop_words = fp.readlines()
    fp.close()
    sw = []
    for i in stop_words:
        sw.append(i.replace('\n', ''))
    line = ''
    for word in words:
        if word not in sw:
            line = line + " " + word

    out = svc.predict(vectorizer.transform([line]))
    logger.info(out)
    lable = newsCH.target_names[out[0]]
    logger.info('预测分类标签为 %s' % lable)
    proba = svc.predict_proba(vectorizer.transform([line]))[0][out[0]]
    logger.info('属于该标签的概率为 %s' % proba)

    return lable


@app.post("/item")
async def item(mod: Mod):
    # # 将数据处理成sklearn要求的格式
    # preprocess('stopwords_cn.txt', 'trainData/')

    # 加载数据集
    newsCH = datasets.load_files('corpus')

    # # 训练模型，并保存
    # train_model(newsCH, 'model/tfidf-vectorizer.pkl', 'model/svc_train_model.m')

    # 调用模型，进行预测
    lable = svc_predict(newsCH, 'model/tfidf-vectorizer.pkl', 'model/svc_train_model.m', mod.doc)

    return lable


if __name__ == '__main__':
   u.run("svmAPI:app", host='localhost', port=8083, reload=True)

