import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import io
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="주성분 분석(PCA) 알고리즘의 개념 및 예시 프로젝트",
    page_icon="🧊",
)

st.title('주성분 분석(PCA) 알고리즘의 개념 및 예시 프로젝트')
st.caption('2022-08-15, 2022 선린인터넷고등학교 20612 양현준 작성')

st.header('PCA란?')
st.markdown(
    '주성분 분석(PCA, Principal Component Analysis)란, 비지도 학습의 차원축소에 속하는 알고리즘으로, 여러 변수를 선형 결합하여 분산이 큰 축을 변수의 수만큼 생성하는 알고리즘입니다. 좀 더 쉽게 말하자면, 어떤 데이터들의 집합에서 가장 크게 해당 데이터를 구분짓는 변수를 찾아서 자료의 차원을 줄이는 알고리즘을 말합니다.'
)
st.subheader('비지도학습과 차원축소는 무엇인가요?')
st.markdown('##### 비지도 학습')
st.markdown('+ 독립변수와 종속변수의 구분이 없는 데이터를 제공하여 학습시키는 머신러닝 방법')
st.markdown('+ 데이터의 \'정답\' 특성에 해당하는 종속변수가 존재하지 않음')
st.markdown('##### 차원축소')
st.markdown(
    '+ 데이터가 가지는 특성의 개수를 \'차원\'이라고 하는데, 데이터의 특성들 중에서 핵심적인 특성만 남기고 나머지 특성들을 제거하는것'
)
st.markdown('차원축소 방법에서 유명한 알고리즘 중 하나가 바로 PCA(주성분 분석) 입니다.')

st.subheader('그럼 PCA를 언제 사용하나요?')
st.markdown(
    '우리가 데이터 분석을 하다보면 독립변수가 많을 때가 있습니다. 독립변수가 매우 많은 경우 고차원의 저주, 다중공선성 문제가 발생할 수 있습니다. 이 두가지 문제를 해결하는 방법 중 하나가 PCA 입니다.'
)
st.markdown('##### 고차원의 저주')
st.markdown(
    '+ 모델이 매우 복잡해져 학습시간이 매우길고, 모델이 일반화되지 못한 과적합 상태가 되어 예측력등의 퍼포먼스가 저하되는 현상')
st.markdown('##### 다중공선성 문제')
st.markdown('+ 각 변수 간에 상관성이 있는 경우, 모델이 제대로 학습되지 못하는 현상')
st.markdown('+ 선형회귀인 경우, 독립변수의 계수가 불안정해짐')
st.markdown('###### 예시')
pcaExampleDf = pd.DataFrame([[1, 3], [2, 5], [3, 8], [4, 9], [5, 10]],
                            columns=('x%d' % i for i in range(1, 3)))
st.table(pcaExampleDf)
fig, ax = plt.subplots()
ax.scatter([1, 2, 3, 4, 5], [3, 5, 8, 9, 10])
st.pyplot(fig)
st.caption('이해를 돕기위한 예시')
st.markdown(
    'x1과 x2, 두 변수는 상관성이 있다고 해봅시다. 이 경우, 위에서 언급했던 다중공선성 문제가 발생할 수 있습니다. 이러면 모델의 설명력이 약해지며 복잡도 역시 증가합니다. 이러한 문제로 두 변수를 한 모델에 같이 사용할 수 없기 때문에 우리는 하나의 변수를 사용하거나, 새로운 축을 찾아야 합니다.'
)
st.markdown(
    '만약 하나의 변수만을 사용할때, 점끼리 값이 겹치는 경우가 생긴다면 정보의 손실이 발생하게 됩니다. 정보 손실을 최소화하면서 변수를 줄이기 위해서는 새로운 축을 찾아야합니다. 이때, 새로운 축은 수학적으로 분산(정보량)을 최대화하는 축이 되며 이 방법을 주성분 분석이라고 합니다. '
)
st.markdown(
    '원래 데이터의 독립변수의 분산을 가장 잘 설명하는 축의 개수를 선정해서 그 축에 따라 변형된 데이터를 배열하면 그 데이터가 주성분이 됩니다.'
)
st.markdown(
    '따라 주성분 분석을 사용하는 경우 원래 데이터에 대한 일부 정보는 잃게 됩니다. 하지만 원래 변수를 몇 개의 주성분으로 축소하면서 생기는 이득이 훨씬 크다고 판단하기 때문에 주성분 분석을 사용합니다.'
)
st.subheader('주의! PCA를 쓰기전 반드시 스케일링을 해주세요')
st.markdown(
    'PCA를 하기 전에는 반드시 데이터 스케일링을 해줘야 합니다. 스케일링을 하지 않는다면 변인이 가진 값의 크기에 따라 설명 가능한 분산량이 왜곡될 수 있습니다. 즉, 데이터의 스케일에 따라 주성분의 설명 가능한 분산량이 달라질 수도 있다는 거죠. 이는 모델의 성능의 저하를 일으킵니다.'
)
st.header('PCA 예시 프로젝트')
st.caption(
    '위 프로젝트에서 사용된 데이터는 2022 선린인터넷고등학교 \'인공지능과 미래사회\' 수업시간에 쓰인 Fish.csv 데이터입니다.'
)
st.markdown('#### 모듈 불러오기')

st.code('''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
''',
        language='python')
st.markdown('프로젝트에 사용될 모듈을 불러옵니다')

st.markdown('#### 데이터 불러오기')

st.code('''fish = pd.read_csv('./Fish.csv')
fish.head(10)
''',
        language='python')
st.markdown('불러온 csv파일을 위에서 10개 줄 까지만 출력합니다.')

fish = pd.read_csv('./Fish.csv')
st.dataframe(fish.head(10))
st.markdown('#### 데이터 전처리')

st.code('''
fish.info()
''', language='python')

buffer = io.StringIO()
fish.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.code('''
fish['Species'] = fish['Species'].str.replace('Bream', '0')
fish['Species'] = fish['Species'].str.replace('Roach', '1')
fish['Species'] = fish['Species'].str.replace('WhiteFish', '2')
fish['Species'] = fish['Species'].str.replace('Parkki', '3')
fish['Species'] = fish['Species'].str.replace('Perch', '4')
fish['Species'] = fish['Species'].str.replace('Pike', '5')
fish['Species'] = fish['Species'].str.replace('Smelt', '6')
fish['Species'] = pd.to_numeric(fish['Species'])
fish.head(10)
''')
st.markdown(
    '첫번째 전처리에서는 Bream을 0, Roach를 1, Whitefish를 2, Parkki를 3, Perch를 4, Pike를 5, Smelt를 6으로 간주하고 replace를 통해 바꿨습니다.'
)

fish['Species'] = fish['Species'].str.replace('Bream', '0')
fish['Species'] = fish['Species'].str.replace('Roach', '1')
fish['Species'] = fish['Species'].str.replace('Whitefish', '2')
fish['Species'] = fish['Species'].str.replace('Parkki', '3')
fish['Species'] = fish['Species'].str.replace('Perch', '4')
fish['Species'] = fish['Species'].str.replace('Pike', '5')
fish['Species'] = fish['Species'].str.replace('Smelt', '6')
fish['Species'] = pd.to_numeric(fish['Species'])
st.dataframe(fish.head(10))

st.code('''
scaler = StandardScaler()
targets = fish['Species']
fish = fish.drop(columns='Species')
scaler.fit(fish)
scaled_fish = scaler.transform(fish)
scaled_fish = pd.DataFrame(scaled_fish,
                           columns=['Weight', 'Length', 'Height', 'Width'])
scaled_fish['Species'] = targets
scaled_fish.head(10))
''',
        language='python')

st.markdown(
    '아까 언급했듯이, PCA를 하기 위해서는 정규화(Normalizaion)를 필수로 해야합니다. 따라서 StandardScaler를 통해 스케일링을 진행했습니다.'
)
scaler = StandardScaler()
targets = fish['Species']
fish = fish.drop(columns='Species')
scaler.fit(fish)
scaled_fish = scaler.transform(fish)
scaled_fish = pd.DataFrame(scaled_fish,
                           columns=['Weight', 'Length', 'Height', 'Width'])
scaled_fish['Species'] = targets
st.dataframe(scaled_fish.head(10))

dim = 2
st.markdown('#### PCA 차원축소')
st.code('''
dim=2
pca = PCA(n_components=dim)
pca.fit(scaled_fish.iloc[:, :-1])
df_pca = pca.transform(scaled_fish.iloc[:, :-1])
df_pca = pd.DataFrame(df_pca, columns = ['components %d'% i for i in range(dim)])
df_pca.head(10)
''',
        language='python')
pca = PCA(n_components=dim)
pca.fit(scaled_fish.iloc[:, :-1])
df_pca = pca.transform(scaled_fish.iloc[:, :-1])
df_pca = pd.DataFrame(df_pca,
                      columns=['components %d' % i for i in range(dim)])
st.dataframe(df_pca.head(10))

st.markdown('원래 있던 4가지의 변수를 ' + str(dim) +
            '개의 주성분으로 차원축소했습니다. 차원축소 진행시 타겟클래스 정보가 포함되지 않게 진행해야 합니다.')

st.markdown('#### PCA 주성분 설명력 출력')
st.markdown(
    '차원 축소된 결과에서 첫 주성분이 가장 데이터 분포에서의 분산에 대한 설명력이 높고, 이후 주성분으로 갈수록 설명력이 낮아지게 됩니다. 위에서 학습된 PCA의 explained_variance_ratio_ 속성을 불러오면 각 주성분에 대한 설명력을 확인할 수 있습니다. 보통 주성분 설명력 합이 80~90%를 넘는 정도까지 활용한다고 합니다.'
)
st.code('''
pca.explained_variance_ratio_
''', language='python')
st.write(pca.explained_variance_ratio_)
st.markdown(
    '첫번째 주성분이 약 86%, 2번째 주성분이 약 10%의 분산 설명력을 가져 두 주성분이 4차원 데이터 분포의 특징중 96%를 설명하고 있습니다.'
)

st.markdown('#### PCA 결과 시각화')
st.code('''
df_pca['Species'] = targets
color_sets = ['orange', 'red', 'green', 'black', 'yellow', 'blue', 'plum']
lb_sets = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
for x in range(7):
    pca_x = df_pca[df_pca['Species'] == x]
    plt.scatter(pca_x['components 0'],
                pca_x['components 1'],
                color=color_sets[x],
                alpha=0.7,
                label=lb_sets[x])
plt.xlabel('components 0')
plt.ylabel('components 1')
plt.legend()
plt.show()
''',
        language='python')
df_pca['Species'] = targets
color_sets = ['orange', 'red', 'green', 'black', 'yellow', 'blue', 'plum']
lb_sets = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
fig1, ax1 = plt.subplots()
for x in range(7):
    pca_x = df_pca[df_pca['Species'] == x]
    ax1.scatter(pca_x['components 0'],
                pca_x['components 1'],
                color=color_sets[x],
                alpha=0.7,
                label=lb_sets[x])
ax1.set_xlabel('components 0')
ax1.set_ylabel('components 1')
ax1.legend()
st.pyplot(fig1)
st.markdown('4개의 특성을 2개의 주성분으로 줄였음에도, 2개의 주성분이 나름 분포 특징을 잘 잡아내고 있습니다.')

st.header('정리 & 소감')
st.markdown(
    '인공지능 소수전공을 들으면서 PCA 알고리즘에 대해 관심이 생겨 프로젝트를 진행하게 되었습니다. 예전에 인공지능과 미래사회 수업시간에 사용되었던 생선 데이터를 이용하여 PCA를 진행해봤습니다. PCA진행 이전에는 총 4개의 독립변수가 있었고, 독립변수가 많아짐에 따라 고차원의 저주와 다중공선성 문제가 발생했습니다. 하지만 PCA알고리즘을 통해서 2개의 주성분으로 요약한 결과, 2개의 주성분으로도 충분히 생선데이터의 분포 특징을 잘 잡아내고 있음을 알 수 있었습니다. 또한 독립변수가 줄어듦에 따라 고차원의 저주 및 다중공선성 문제 역시 줄어들게 되어 이득을 보았습니다.'
)
st.markdown(
    '인공지능과 미래사회 1학기 프로젝트에서 자료조사 및 데이터 전처리를 담당하였을때, 저는 무조건 많은 독립변수를 만들려고 하였습니다. 하지만 이번에 PCA 알고리즘을 탐구하고 간단한 프로젝트를 진행하면서 독립변수가 너무 많으면 오히려 독이 될 수도 있다는 것을 알게 되었습니다. 또한 차원축소를 왜 진행하는지에 대해서도 알게 되었고, 대표적인 차원축소 기법인 PCA 이외에도 LDA, SVD등 다양한 차원축소 기법들에 대해서도 공부할 수 있었습니다.'
)
st.markdown(
    '또한, Streamlit 이라는 데이터 분석에 특화된 파이썬 기반 오픈소스 웹 프레임워크를 사용하여 프로젝트 결과를 웹페이지로 제작하여 배포하였습니다. 기존 방식인 주피터 노트북을 사용하지 않고 진행한 프로젝트를 웹으로 배포한 경험 역시 재미있었습니다.'
)

st.header('참고자료')
st.markdown('+ 2022 선린인터넷고등학교 인공지능과 미래사회 수업자료')
st.markdown('+ [파이썬 사이킷런PCA 구현 및 시각화 예제](https://jimmy-ai.tistory.com/128)')
st.markdown(
    '+ [주성분 분석의 개념적 이해](https://everyday-tech.tistory.com/entry/%EC%A3%BC%EC%84%B1%EB%B6%84-%EB%B6%84%EC%84%9DPCA)'
)
