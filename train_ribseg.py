"""
Author: Benny
Date: Nov 2019
"""
import argparse #사용자가 입력한 명령행의 인자를 파싱한 후 인자 값에 다라 적당한 동작을 수행. 이처럼 명령행의 인자를 파싱할때 사용하는 모듈이 바로 argparse.
#파싱: 프로그램을 compile하는 과정에서 특정 프로그래밍 언어가 제시하는 문법을 잘 지켜서 작성하였는지 compiler가 검사하는 것
#파싱: 주어진 정보를 내가 원하는 대로 가공하여 서버에서 원하는 때 불러올 수 있도록 하는 것, 어떤 data를 원하는 form으로 만들어 내는 것
import os
from data_utils.RibFracDataLoader_1cls import PartNormalDataset
import data_utils.data_trans as d_utils
import torchvision
import torch
import datetime #날짜와 시간을 조작하는 클래스 제공
import logging #현재 프로그램이 어떤 상태를 가지고 있는지, 외부 출력을 하게 만들어서 개발자 등이 눈으로 직접 확인하는 것.
from pathlib import Path
import sys 
import importlib
import shutil #shutil 모듈은 파일 모음에 대한 여러가지 고수준 연산을 제공합니다. 특히, 파일 복사와 삭제를 지원하는 함수가 제공됩니다.
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #절대 경로 반환 #__file__은 python의 예약어로, 실행되는 스크립트 파일명을 나타낸다.
#aspath의 argument는 해당 경로에 실제로 파일이 존재해야만 하는 것은 아니다. 임의의 문자열을 넣어도 해당 경로를 반환해준다.
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models')) #경로 합치기 # https://devbruce.github.io/python/py-39-path+function/


# seg_classes = {'rib':[0,1]}
# we adapted the PointNet++ code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git, 
# to keep the consistency, we simply let the number of classes be the same as the source code.
# For experiments, the only classes used will be 'rib': 0 ,1.
#[]=> array : 배열의 원소에 접근할때 사용
#()=> tuple : 딕셔너리와 비슷하지만 튜플은 이미 생성된 원소를 제거하거나, 변경할 수 없다. 또 튜플은 원소의 타입이 같을때 사용 
#{}=> dictionary : key에 대응하는 value 을 할당하거나 value에 접근할때 사용
seg_classes = {'rib':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],'Earphone': [44, 45], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [29], 'Laptop': [28 ], 'Cap': [26], 'Skateboard': [46],
                            'Mug': [36], 'Guitar': [39, 40], 'Bag': [27], 'Lamp': [25],
                            'Table': [47], 'Airplane': [48], 'Pistol': [38],
                            'Chair': [37], 'Knife': [49]}


seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table} # 빈 딕셔너리 생성 시 {} 사용
for cat in seg_classes.keys(): # for in 함수
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    #torch.eye() : 사선 방향이 1인 a x a 텐서 생성, 대각선에 1이 있고 다른 곳에 0이 있는 2차원 텐서를 반환
    #.data : Variable에서 값을 얻는 attribute # cpu(): GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사하는 method
    # .numpy() tensor를 numpy로 변환하여 반환. 이때 저장공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경된다.
    if (y.is_cuda):
        return new_y.cuda() #cuda: nvidia에서 개발한 gpu 개발 툴 : 컴퓨터 연산을 위해
    return new_y


def parse_args(): #인자: 매개변수
    parser = argparse.ArgumentParser('Model') #argument parser 객체 생성= argument parser 객체는 명령행을 파이썬 데이터 형으로 파싱하는 데 필요한 모든 정보를 생성
    #argument parser에 프로그램 인자에 대한 정보를 채우려면 add_argument() 메서드를 호출하면 된다.=> 이 호출은 argument parser에게 명령행의 문자열을 객체로 변환하는 방법을 알려준다
    #이 정보는 저장되고 parse_args()가 호출될 때 사용된다.
    #argparse 라이브러리를 사용하여 model, batch_size, epoch 등 설정-> 프로그램에 필요한 인자들을 정의
    # 인자의 앞에 --,-가 붙어 있으면 optional 인자(선택형 인자), 붙어있지 않으면 positional 인자(위치형 인자)이다. 
    #위치형 인자는 필수적으로 입력해야하는 인자이며ㅡ 선택형 인자도 required=True를 통해 필수로 입력하게 끔 지정할 수 있다.
    #typedml default값은 str이며, default=를 통해 사용자가 옵션을 주지 않았을 때 기본적으로 들어가는 값을 지정하는 곳이다.
    #help: 인자에대한 설명을 쓴다.
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=30000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args() #인자 파싱하기 => argument parser는 parse_args()메서드를 통해 인자를 파싱합니다. 이 메서드는 명령행을 검사하고 각 인자를 적절한 형으로 변환한다음
  #적절한 액션을 호출합니다.

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER''' #gpu에 메모리 할당
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True) #폴더 생성경로에 폴더가 없을 경우 자동으로 생성
    experiment_dir = experiment_dir.joinpath('part_seg') #경로 이동: 디렉토리 트리 내에서 이동할 수 있도록 어떤 path 객체를 기준으로한 다른 디렉토리나 파일을 표시하는 pat객체 생성
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/') 
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model") #자신만의 특정한 로거를 따로 사용
    logger.setLevel(logging.INFO) #setLevel 메소드를 통해서 INFO 레벨 이상은 출력하도록 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')# 이 메세지가 언제쓰여졌는지, 어떤 모듈에서 쓰여졌는지 등 기타 정보를 같이 출력할때
    #asctime: 시간 , name:로거이름, levelname:로깅레벨, message:메세지
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))#logging.FileHandler 클래스를 통해 객체를 만들어서 나의 로거에 추가
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = './data/pn/'

    transforms = torchvision.transforms.Compose( #다양한 이미지 변환 기능들을 제공해준다. 
   #data_utils에 있는 코드
      [
            d_utils.PointcloudToTensor(), 
            # d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='trainval',transforms=transforms, normal_channel=args.normal)#transforms
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=0) #shuffle=true 무작위 샘플링
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='test', transforms=None,normal_channel=args.normal) #parser: npoint, normal transforms(x)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=0) 
    #batch_size= 배치의 크기 : 데이터셋에 50개의 데이터가 있고 batch_size가 10이라면 5번의 iteration만 지나면 모든 데이터를 볼 수 있다.
    #shuffle: 데이터를 DataLoader에서 섞어서 사용하겠는지를 설정할 수 있다.
    #num_workers: 데이터 로딩에 사용하는 subprocess개수이다.=> 기본 값이 0인데 이는 data가 mainprocess로 불러오는 것을 의미한다.
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16 #1 #최종 출력 클래스의 크기
    num_part = 50 #2
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)#모듈을 임포트합니다. import_module()함수는 importlib.__import__() 주위를 감싸는 단순화 wrapper 역할
    shutil.copy('models/%s.py' % args.model, str(experiment_dir)) #shutil 모듈은 파일 모음에 대한 여러가지 고수준 연산을 제공합니다. 특히, 파일 복사와 삭제를 지원하는 함수가 제공됩니다.
    shutil.copy('models/pointnet_util.py', str(experiment_dir)) #기능: 파일을 복사한다. shutil.copy(src파일 경로, dest 파일(or 폴더)경로) #src 원본 파일 dest 대상파일

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda() #TMI: 정의하는 느낌
    criterion = MODEL.get_loss().cuda()

#가중치 초기화 #모델 초기화
    def weights_init(m):
        classname = m.__class__.__name__# 클래스 이름을 참조한다. 클래스명 m (ex 클래스명.__name__)
        if classname.find('Conv2d') != -1: #find 함수: "찾을 문자" 혹은 "찾을 문자열"이 존재하는지 확인하고, 찾는 문자가 존재한다면 해당 위치의 index를 반환/ 존재x -1반환
            torch.nn.init.xavier_normal_(m.weight.data) #xavier 초기화는 고정된 표준편차를 사용하지 않다는 특징이 있다.
            #이전 은닉층의 노드수(fan_in)과 현재 은닉층의 노드(fan_out)을 고려하여 만들어진다. 활성값이 고르게 분포한다
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data) #xavier 초기화는 고정된 표준편차를 사용하지 않다는 특징이 있다.
            #이전 은닉층의 노드수(fan_in)과 현재 은닉층의 노드(fan_out)을 고려하여 만들어진다. 활성값이 고르게 분포한다.
            torch.nn.init.constant_(m.bias.data, 0.0) #torch.nn.init.constant_(tensor,val) tensor: n차원 val: the value to fill the tensor with
#try except: 예외처리를 하려면 다음과 같이 try에 실행할 코드를 넣고 except에 예외가 발생했을 때 처리하는 코드를 넣는다.
    try: 
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth') #torch.load: pickle을 사용하여 저장된 객체 파일들을 역직렬화하여 메모리에 올립니다.
      #pickle 모듈은 python 객체 구조를 직렬화 및 역직렬화하기 위한 바이너리 프로토콜을 구현한다. #pytorch에서 학습한 모델 저장 및 불러오기
      
      #pytorch의 모델은 직렬화와 역직렬화를 통해 객체를 저장하고 불러올 수 있다.
      #모델을 저장하는 방법은 python의 피클을 활용하여  파있너 객체 구조를 바이너리 프로토콜로 직렬화 합니다.
      #모델을 불러오는 방법은 저장된 객체 파일을 역직렬화하여 현재 프로세스의 메모리에 업로드합니다.
    
        start_epoch = checkpoint['epoch']  #checkpoint는 모델이 사용한 모든 매개변수의 정확한 값을 캡처한다.
        classifier.load_state_dict(checkpoint['model_state_dict'])#모델 불러오기
        log_string('Use pretrain model')
    except: #TMI 모델을 가지고 있지 않으므로 시작할 수 없어서 START_EPOCH=0
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999), #그레디언트와 그 제곱의 실행 평균을 계산하는데 사용되는 계수(기본값) #기울기 평균을 계산하는 매개변수
            eps=1e-08, #수치 안정성을 개선하기 위해 분모에 추가된 용어(기본값) 
            weight_decay=args.decay_rate #가중치 감소=> weight_decay의 값이 커질수록 가중치 값이 작아지고, 오버피팅을 해소할 수 있다. weight_decay값을 너무 크게하면 언더피팅 발생
        )
    else: #TMI: Adam이 안되면 SGD 최적화
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum): # 주어진 인스턴스가 특정 클래스/ 데이터 타입인지 검사해주는 함수
      #isinstance(확인하고자하는 데이터 값, 확인하고자 하는 데이터 타입)=>isinstance(인스턴스, 데이터나 클래스타입)
      #첫번째 매개변수: 확인하고자하는 데이터의 값, 객체, 인스턴스 #두 번째 매개변수 : 확인하고자 하는 데이터타입, 클래스 #반환값: 인스턴스와 타입이 같으면 True 아니면 False
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d): # 조건들 중 하나만 True여도 if문 코드가 실행
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5 
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5 #0보다 크거나 같은 float 값. 업데이트마다 적용되는 학습률의 감소율
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch)) #epoch값 출력 시작
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP) #learning rate 
        log_string('Learning rate:%f' % lr) #learning rate 값 출력
        for param_group in optimizer.param_groups: #이 함수를 호출할 때 사용하는 optimizer와 대응하는 epoch를 입력하고 args.lr를 초기화하는 학습율로도 제시
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))#momentu=0.1*(0.5제곱)
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))#lambda()함수는  사용자 정의 함수를 문법에 맞추어 작성하는 것보다 간단하게 해결할 수 있는 함수
        #dataframe명.apply(lambda x :x['칼럼명']들의 조건식 if x['칼럼명']들의 조건식

        '''learning one epoch''' 
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            # print(points.shape,label.shape,target.shape)
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3]) # : =모든 성분을 추출 3차원 array[행,열] #provide.py에 함수: random_scale_point_cloud
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3]) #provide.py에 함수: shift_point_cloud
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad() #pytorch에서는 gradients값들을 추후에 backward를 해줄 때 계속 더해주기 때문에 항상 backpropagation을 하기 전에 gradients를 zero로 만들어주고 시작
            classifier = classifier.train()
            
            # print(points.size())
            # print('label:',label,label.size())
            a=to_categorical(label, num_classes) #num_classes=16 최종출력 클래스 크기 #to_categorical: one-hot인코딩을 해주는 함수 10진 정수 형식을 특수한 2진 바이너리 형식 변환
            #to_categorical 함수는 입력받은 n크기의 1차원 정수 배열을 (n,k)2차원 배열로 변경. 이 배열의 두 번째 차원의 인덱스가 클래스값을 의미
            # print('ssss:',a,a.size())
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part) #.contiguous()메서드는 다음과 같이 이러한 a와 같은 비연속적인 텐서를 연속적으로 만들어주는 역할
            #non-contiguous Tensor 객체의 경우 주소값 연속성이 불변인 것을 해결해주는 contiguous()를 사용하여 새로운 메모리 공간에 데이터를 복사하여 
            #주소값 연속성을 가변적이게 만들어주는 것.-> contiguous()결과가 원본과 다른 새로운 주소로 할당된것을 확인할 수 있다.
            #contiguous함수로 새로운 메모리에 할당하여 contiguous Tensor로 변경하면 주소값 재배열이 가능하다.
            #num_part=50
            #view(): view는 기존의 데이터와 같은 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게한다. 그래서 contiguous해야하만 동작하며, 아닌 경우 에러 발생
            # print('seg_pred',seg_pred.shape)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum() #GPU 메모리에 올려져있는 tensor를  cpu메모리로 복사하는 method #eq 같다(비교 메서드)
            mean_correct.append(correct.item() / (args.batch_size * args.npoint)) #이전 mean_correct=> mean_correct=[] #items()함수를 사용하면 딕셔너리에 있는 키와 값들의
            #쌍을 얻을 수 있다.
            loss = criterion(seg_pred, target, trans_feat) #그 전 코드  criterion = MODEL.get_loss().cuda() #trans_feat 13줄 전에 언급 
            loss.backward() #error를 backpropagation을 하기위해 사용. 기존에 계산된 변화도의 값을 누적시키고 싶지 않다면 기존에 계산된 변화도를 0으로 만드는 작업 필요
            #backward함수는 어떤 스칼라 값에 대한 출력텐서의 변화도를 전달받고, 동일한 스칼라 값에 대한 입력 텐서의 변화도를 계산
            optimizer.step() #변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 매개변수를 조정
            #결론: backward로 미분하여 손실함수에 끼친 영향력(변화량)을 구하고 optimizer.step을 통해 손실함수를 최적화하도록 파라미터를 업데이트
        train_instance_acc = np.mean(mean_correct)#산술 평균
        log_string('Train accuracy is: %.5f' % train_instance_acc)
#------------------Train 정확도 출력--------------------------#
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys(): #seg_class의 키값들, 위에서도 반복했던 코드
                for label in seg_classes[cat]: 
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
              #enumerate() 함수는 인자로 넘어온 목록을 기준으로 인덱스와 원소를 차례대로 접근하게 해주는 반복자(iterator) 객체를 반환해주는 함수
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda() #Train위에도 있는 부분
                points = points.transpose(2, 1)#Train위에도 있는 부분
                classifier = classifier.eval() #Train 위에는 classifier.train()
                seg_pred, _ = classifier(points, to_categorical(label, num_classes)) #train부분이랑 다른점  ,_부분 위에는 trans_feat
                cur_pred_val = seg_pred.cpu().data.numpy()# train 위에는 sum()
                cur_pred_val_logits = cur_pred_val 
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32) #np.zeros=>zeros()함수는 0으로 초기화된 shape 차원의 ndarray 배열 객체를 반환
                #numpy zeros method는 0으로만 채워진 array를 생성 #astype()로 데이터형 dtype을 변경(타입변경)
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size): #cur_batch_size =points.size()
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        print('Dice:',loss)
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

