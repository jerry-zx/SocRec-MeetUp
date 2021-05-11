import numpy as np
import os
from collections import defaultdict
import json
import math
Mlist=[]
Mdic={}
Edic={}
Elist=[]
MEdic={} #key是两个用户的用户名，vlaue是边权。建立pair的索引
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



class member:
    def __init__(self,id):
        self.id=id
        self.topic=[]
        self.yexe=[]
        self.nexe=[]
        self.mexe=[]
    def gettopic(self,topic):
        self.topic.append(topic)
    def getyexe(self,yexe):
        self.yexe.append(yexe)
    def getnexe(self,nexe):
        self.nexe.append(nexe)
    def getmexe(self,mexe):
        self.mexe.append(mexe)


class group:
    def __init__(self):
        self.id=None;
        self.smember = None;
        self.topic = None;
    def getinfo(self,id,semember,topic):
        self.id = id;
        self.smember = semember;
        self.topic = topic;


class event:
    def __init__(self,id,time):
        self.id = id
        self.time = time
        self.ymembers = []
        self.nmembers = []
        self.mmembers = []
    def getym(self,m):
        self.ymembers.append(m)
    def getnm(self,m):
        self.nmembers.append(m)
    def getmm(self,m):
        self.mmembers.append(m)




def setexe():
    for i in range(0,481):
        print("第",i+1,"个文件")
        if(i%10==0): #采样文件
            file = os.path.join(os.getcwd(), "GroupEvent/G{0}.txt".format(i))
            f = open(file)
            for i,line in enumerate(f):
                if line.strip() == "":
                    # doing something
                    pass
                elif line in ['\n', '\r\n']:
                    # doing something
                    pass
                else:
                    for s in line.split():
                        if (s[0] == "E"):
                            j=i
                            info=line.split()
                            exe=info[0]
                            time=info[2]
                            e=event(exe,time)
                            Edic[exe]=e
                            Elist.append(exe)
                            break
                        elif(i-j==2):#yes用户
                            for s in line.split():
                                if(s!="null"):
                                    Mdic[s].getyexe(exe)
                                    Edic[Elist[-1]].getym(s)
                        elif(i-j==3):#no
                            for s in line.split():
                                if(s!="null"):
                                    Mdic[s].getnexe(exe)
                                    Edic[Elist[-1]].getnm(s)
                        elif(i-j==4):#maybe
                            for s in line.split():
                                if(s!="null"):
                                    Mdic[s].getmexe(exe)
                                    Edic[Elist[-1]].getmm(s)


def com(am,bm,type):
    same=0
    if(type=="topic"):
        alist=am.topic
        blist=bm.topic
    elif(type=="yexe"):
        alist=am.yexe
        blist=bm.yexe
    elif(type=="nexe"):
        alist=am.nexe
        blist=bm.nexe
    elif(type=="mexe"):
        alist=am.mexe
        blist=bm.mexe
    for t1 in alist:
        for t2 in blist:
            if(t1==t2):
                same+=1
    return same



def fillneighbor(neighbor):#补全孤立节点信息
    print(len(neighbor))
    with open("./mem2idx.json", 'r', encoding='utf-8') as f_in:
        memberlist = json.loads(f_in.readline())
    i=0
    with open("./mem2idx.json", 'r', encoding='utf-8') as f_in:
        member_list = json.loads(f_in.readline())
    for member in memberlist:
        if(member_list[member] not in neighbor.keys()):
            i+=1
            emp=[]
            neighbor[member_list[member]]=emp
    print("共",i,"个孤立节点")
    print(len(neighbor))
    return  neighbor

def time_event_getneighbor(i):
    print("进入建邻居")
    with open("./mem2idx.json", 'r', encoding='utf-8') as f_in:
        member_list = json.loads(f_in.readline())
    neighbor=defaultdict(list)
    for lis in MEdic.values():
        neighbor[member_list[lis[0]]].append([member_list[lis[1]],lis[2]])
        neighbor[member_list[lis[1]]].append([member_list[lis[0]],lis[2]])
    for lis in neighbor.values():
        lis.sort(key=lambda x:x[1])
    neighbor = fillneighbor(neighbor)
    path = os.path.join(os.getcwd(), "./time_event_neighbor{0}.json".format(i))
    with open(path, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(neighbor, ensure_ascii=False, cls=NpEncoder))
    print("建完邻居",path)


#内部调用
#----------------------------
#外部调用

def getmember():
    f = open("./MemberTopic.txt")
    for line in f:
        if line.strip() == "":
            # doing something
            pass
        elif line in ['\n', '\r\n']:
            # doing something
            pass
        else:
            for s in line.split():
                if(s[0]=="M"):
                    m=member(s)
                    Mdic[s]=m
                    Mlist.append(m.id)
                else:
                    Mdic[Mlist[-1]].gettopic(s)
    np.save('./memberlist',Mlist)



def graph(type):
    if(type!="topic"):
        setexe()
    print("开始创建图")
    print("一共",len(Mlist),"个节点")
    graph=[]
    for i in range(len(Mlist)):
        am=Mdic[Mlist[i]]
        for j in range(i+1,len(Mlist)):
            bm=Mdic[Mlist[j]]
            same=com(am,bm,type)
            if(same!=0):
                pair=[]
                pair.append(i)
                pair.append(j)
                pair.append(same)
                graph.append(pair)
    if(type=="topic"):
        np.save('./tgragh',np.array(graph))
    elif(type=="yexe"):
        np.save('./yexegraph', np.array(graph))
    elif(type=="nexe"):
        np.save('./nexegraph', np.array(graph))
    elif(type=="mexe"):
        np.save('./mexegraph', np.array(graph))


def getneighbor(type):
    memberlist=np.load("./memberlist.npy")
    with open("./mem2idx.json", 'r', encoding='utf-8') as f_in:
        member_list = json.loads(f_in.readline())
    if(type=="topic"):
        graph=np.load('./tgragh.npy')
        path="./tneighbor.json"
    elif(type=="yexe"):
        graph=np.load('./yexegraph.npy')
        path="./yexeneighbor.json"

    elif(type=="nexe"):
        graph=np.load('./nexegraph.npy')
        path="./nexeneighbor.json"

    elif(type=="mexe"):
        graph=np.load('./mexegraph.npy')
        path="./mexeneighbor.json"



    neighbor=defaultdict(list)
    print("共",len(graph))
    for i,pair in  enumerate(graph):
        # if(i%100==0): #均匀采样
        # if(pair[2]>=10): #阈值
            print(pair[0],pair[1])
            neighbor[member_list[memberlist[pair[0]]]].append([member_list[memberlist[pair[1]]],pair[2]])
            neighbor[member_list[memberlist[pair[1]]]].append([member_list[memberlist[pair[0]]],pair[2]])
    for lis in neighbor.values():
        lis.sort(key=lambda x:x[1])
    neighbor = fillneighbor(neighbor)

    # if(type=="topic"):
    #     np.save("./threshold8_tneighbor",neighbor)
    # else:
    with open(path, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(neighbor, ensure_ascii=False,cls=NpEncoder))

def get():
    # print("开始读取")
    with open("./10tneighbor_ratio.json", 'r', encoding='utf-8') as f_in:
        neighbor = json.loads(f_in.readline())
    print(len(neighbor))
    # memberlist=np.load("./memberlist.npy")
    # print(len(memberlist))
    with open("./mem2idx.json", 'r', encoding='utf-8') as f_in:
        member_list = json.loads(f_in.readline())
    print(len(member_list))
    # with open("./tneighbor.json", 'r', encoding='utf-8') as f_in:
    #     neighbor = json.loads(f_in.readline())
    # print("读取完成")

def getpair(n1,n2):
    if(n1>n2):
        return n1+n2
    else:
        return n2+n1

def presenttneighbor(ratio): #ratio[0,1]
    neighbor=np.load("./10tneighbor.npy",allow_pickle=True).item()
    for m in neighbor.keys():
        print("开始长度", len(neighbor[m]))
        num=math.ceil(-1*len(neighbor[m])*ratio)
        if(num>-10):
            num==max(-10,-1*len(neighbor[m]))
        neighbor[m]=neighbor[m][num:]
        print("之后长度",len(neighbor[m]))
    with open("./10tneighbor_ratio.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(neighbor, ensure_ascii=False, cls=NpEncoder))

def thresholdtneighbor(threshold):
    neighbor=np.load("./10tneighbor.npy",allow_pickle=True).item()
    for m in neighbor.keys():
        print("跳出寻找")
        for i in range(len(neighbor[m])):
            print("寻找")
            clear=True
            if(neighbor[m][i][1]>=threshold):
                print("进入截断")
                neighbor[m]=neighbor[m][i:]
                clear=False
                break
        if(clear):
            neighbor[m]=[]
    with open("./10tneighbor_threshold.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(neighbor, ensure_ascii=False, cls=NpEncoder))


def time_event_graph(wy,wn,wm,seg): #生成pair的方式与之前不同，这里是查找event，如果还是以用户来进行查找就没有时间序列，是乱的。但是用户可以一次性完整建立两个用户的边，
    #如果以活动查找，就需要存储用户piar的字典，后面其他活动可能还会访问到该pair
    getmember()
    setexe()
    Elist.sort(key=lambda x: x[1])#根据时间戳对活动排序
    print("总的Elist长度：",len(Elist))
    length=math.ceil(len(Elist)/seg);
    print("length",length)
    for idx,e in enumerate(Elist):
        # print("活动idx",idx,"活动time",Edic[e].time,"活动list",Edic[e].ymembers,Edic[e].nmembers,Edic[e].mmembers)
        print("活动idx",idx,"活动time",Edic[e].time)

        for i in range(len(Edic[e].ymembers)):
            for j in range(i + 1, len(Edic[e].ymembers)):
                # print(i,j)
                pairkey=getpair(Edic[e].ymembers[i],Edic[e].ymembers[j])

                # print(pairkey)
                if(pairkey not in MEdic): #不存在该边
                    MEdic[pairkey]=[Edic[e].ymembers[i],Edic[e].ymembers[j],wy]
                else:
                    MEdic[pairkey][2]+=wy  #更新边权
        for i in range(len(Edic[e].nmembers)):
            for j in range(i + 1, len(Edic[e].nmembers)):
                # print(i,j)

                pairkey=getpair(Edic[e].nmembers[i],Edic[e].nmembers[j])
                # print(pairkey)
                if(pairkey not in MEdic): #不存在该边
                    MEdic[pairkey]=[Edic[e].nmembers[i],Edic[e].nmembers[j],wn]
                else:
                    MEdic[pairkey][2]+=wn  #更新边权
        for i in range(len(Edic[e].mmembers)):
            for j in range(i + 1, len(Edic[e].mmembers)):
                # print(i,j)
                pairkey=getpair(Edic[e].mmembers[i],Edic[e].mmembers[j])
                # print(pairkey)
                if(pairkey not in MEdic): #不存在该边
                    print("这里")
                    MEdic[pairkey]=[Edic[e].mmembers[i],Edic[e].mmembers[j],wm]
                else:
                    print("这里")
                    MEdic[pairkey][2]+=wm  #更新边权
        if(idx%length==0 or idx==len(Elist)-1):  #达到一个分段
            print(idx)
            print(idx/length)
            time_event_getneighbor(idx/length)





if __name__ == '__main__':
    # getmember()
    # graph("mexe")
    # getneighbor("mexe")
    # presenttneighbor(0.3)
    # thresholdtneighbor(4)
    # get();
    time_event_graph(2,0.5,0.5,10)
