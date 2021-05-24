# Faster Notes with Python
## Transcribing pdf slides with OCR and python


### A pipeline for taking notes

Usually when I take notes about a lecture, or video, it usually involves a set of pdf slides that I follow along with a video (or not), the process traditionally involves writing a few notes about what is being said and I often find myself copying some of the points made on the actualy pdf slide. Recently I have been experimenting with trying to automate this process by using python to automatically transcribe the pdf slides so that I can directly manipulate their content in a markdown file to avoid having to copy and paste or literally just copy what is in the pdf slide.

In this article I will show you how to automate the process of transcribing pdf slides to text using OCR.

---

### Why not use a traditional pdf to text tool?

The reason why I choose not to use traditional pdf to text tools is that I found that they give more issues then solve them. I tried using traditional python packages like `pdf2text` but they raise so many issues to quickly transcribe the notes that I decided to play around with a little bit of object detection and OCR. 

The neat thing about this approach is that once you get it right, it has a broad range of applications from digitizing handwritten notes to all sorts of image to text problems. 

---


### Steps
The steps to follow will be: 
1. Get the images
2. Detect the text in the images as bounding boxes
3. Feed each detected bounding box to a text recognizer
4. Showcase example outputs

I want to note here that in this particular case I mostly adapted code from this [repository](https://github.com/courao/ocr.pytorch) which I found to be as simple and to work as well as I needed.

Now, let's go through each one by one.

---


## 1. Get the images
To get the images I will use a pdf of introduction to Reinforcement Learning and write some code to get each slide as a png image.



```python
pdf_path = "/home/lucassoares/Desktop/Notes/intro_RL_Lecture1.pdf"
from pdf2image import convert_from_path
from pdf2image.exceptions import (
 PDFInfoNotInstalledError,
 PDFPageCountError,
 PDFSyntaxError
)

images = convert_from_path(pdf_path)

for i, image in enumerate(images):
    fname = "image" + str(i) + ".png"
    image.save(fname, "PNG")
```

Now that I have all of the images:

![image.png](Faster%20Notes%20with%20Python_files/markdown_12_attachment_0_0.png)

Let's run text detection on each.

## 2. Detect the text in the images as bounding boxes &

## 3. Feed each detected bounding box to a text recognizer


To do that we will use a text detector from this [ocr.pytorch repository](https://github.com/courao/ocr.pytorch).


```python
# adapted from this source: https://github.com/courao/ocr.pytorch
%load_ext autoreload
%autoreload 2
import os
from ocr import ocr
import time
import shutil
import numpy as np
import pathlib
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pytesseract


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed

image_files = glob('./input_images/*.*')
result_dir = './output_images_with_boxes/'

# If the output folder exists we will remove it and redo it.
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

for image_file in sorted(image_files):
    t = time.time()
    result, image_framed = single_pic_proc(image_file) # detecting and recognizing the text
    filename = pathlib.Path(image_file).name
    output_file = os.path.join(result_dir, image_file.split('/')[-1])
    txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt')
    print(txt_file)
    txt_f = open(txt_file, 'w')
    Image.fromarray(image_framed).save(output_file)
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print("\nRecognition Result:\n")
    for key in result:
        print(result[key][1])
        txt_f.write(result[key][1]+'\n')
    txt_f.close()
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    ./output_images_with_boxes/image0.txt
    Mission complete, it took 0.238s
    
    Recognition Result:
    
    ecture
    IntroductiontoReinforcement
    Learning
    DavidSilver
    ./output_images_with_boxes/image1.txt
    Mission complete, it took 0.274s
    
    Recognition Result:
    
    Outline
    Admin
    About Reinforcement Learning
    TheReinforcementLearning Problem→
    Inside An RL Agent
    Problems withinReinforcement Learning
    ./output_images_with_boxes/image10.txt
    Mission complete, it took 0.168s
    
    Recognition Result:
    
    -AboutRL
    Bipedal Robots
    ./output_images_with_boxes/image11.txt
    Mission complete, it took 0.168s
    
    Recognition Result:
    
    -AboutRL
    Atari
    ./output_images_with_boxes/image12.txt
    Mission complete, it took 0.487s
    
    Recognition Result:
    
    -TheRL Problem
    —Reward
    Rewards
    reward R. is a scalar feedback signal
    Indicates how well agent is doing at step
    Theagentsjiobistomaximisecumuativereward
    Reinforcement learning is based on thereward hypothesis
    Definition(Reward Hvypothesis)
    Agoascanbe describedbv the maximisation of expected
    cumulativereward
    Doyou agree withthis statement?
    ./output_images_with_boxes/image13.txt
    Mission complete, it took 0.686s
    
    Recognition Result:
    
    -TheRL Probiem
    Reard
    Exampes of Rewards
    helicopter
    Fly stunt manoeuvres in
    +ve rewardfor following desired trajectory
    -verewardfor crashing
    Defeat the worldchampion atBackgammon
    -ve rewardfor winning/losing agame
    Manage an investment portfolio
    +verewardfor each Sinbank
    Controlapower station
    tverewardfor producingpower
    -ve rewardfor exceeding safetv thresholds
    Make
    humanoidrobot walk
    +verewardfor forward motion
    -ve rewardfor falling over
    Play many different Atari games better than humans
    t/-verewardfor increasing/decreasing score
    ./output_images_with_boxes/image14.txt
    Mission complete, it took 0.573s
    
    Recognition Result:
    
    -TheRL Probiem
    Reward
    SequentialDecision Making
    Goal:seect actions to maximise total future reward
    Actions mav havelongterm conseauences
    Rewardmaybedelayed
    may be betterto sacrifice immediaterewardtogain more
    longtermreward
    Examples:
    Afinancial investment(mav take monthstomature)
    Refuellingahelicopter(might prevent acrash in several hours)
    Blockingopponent moves(might help winningchances many
    movesfromnow)
    ./output_images_with_boxes/image15.txt
    Mission complete, it took 0.270s
    
    Recognition Result:
    
    -TheRL Probem
    Enviromments
    Agent andEnvironment
    action
    observation
    reward
    ./output_images_with_boxes/image16.txt
    Mission complete, it took 0.463s
    
    Recognition Result:
    
    -TheRL Probem
    Enviromments
    Agent andEnvironment
    observation
    action
    At each steptthe agent:
    Executes actionA.
    Receives observationO.
    Receives scalar rewardR.
    Theenvironment:
    Receives action A.
    Emits observation Oc1
    Emits scalar rewardR.L1
    tincrements at env.step
    ./output_images_with_boxes/image17.txt
    Mission complete, it took 0.558s
    
    Recognition Result:
    
    -TheRL Probilem
    LState
    History and State
    The historyis the sequence ofobservations,actions,rewards
    A=0.凡,A…A,0R
    i.e.allobservable variables uptotime
    robot or embodied agent
    i.e.the sensorimotor streamof
    What happens next depends on the history:
    The agent selects actions
    The environment selects observations/rewards
    State isthe information usedto determine what happensnext
    Formallv state is afunction ofthe history:
    f(A)
    ./output_images_with_boxes/image18.txt
    Mission complete, it took 0.496s
    
    Recognition Result:
    
    -TheRL Probiem
    LState
    Environment State
    Theenvironment state Sgis
    observatior
    theenvironment sprivate
    representation
    i.e.whateverdatathe
    emvironment usestopickthe
    next observation/reward
    Theenvironment state isnot
    usually visibletotheagent
    EvenifSgisvisible,it may
    containirrelevant
    information
    emvironmemtstateS
    ./output_images_with_boxes/image19.txt
    Mission complete, it took 0.570s
    
    Recognition Result:
    
    -The RL Problem
    -State
    Agent State
    The agent state Sisthe
    agemtstate
    agemt s internal
    representation
    observatiom
    i.e.whatever information
    the agent usestopick the
    next action
    ie.itistheinformation
    usedbvreinforcement
    learning algorithms
    tcanbe anyfunction of
    history
    f(A)
    ./output_images_with_boxes/image2.txt
    Mission complete, it took 0.457s
    
    Recognition Result:
    
    -Admin
    Class Iformation
    Thursdays9:30to1l:00am
    Website:
    http://www.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html
    Group:
    http://groups.google.com/group/csmladvanced-topics
    Contact me:d.silver@cs.ucl.ac.uk
    ./output_images_with_boxes/image20.txt
    Mission complete, it took 0.647s
    
    Recognition Result:
    
    -The RL Problem
    LState
    InformationState
    An information state(a.ka.Markov state)contains all usefu
    informationfromthehistorv.
    Definition
    A state S.is Markovif and onlyif
    P[S.11|S=P[S.1|S.…S
    “Thefutureisindependent of the past given the present
    Ht1:oo
    known,the history maybethrownaway
    Oncethestate
    i.e.The state is asufficient statistic ofthefuture
    The environment state Sg is Markov
    Thehistomy H.is Markov
    ./output_images_with_boxes/image21.txt
    Mission complete, it took 0.426s
    
    Recognition Result:
    
    -TheRL Probiem
    State
    Rat Example
    LL PRESS
    EyER
    FoR
    FoOD
    面
    last 3items in sequence?
    What if agent state
    countsfor lights.bellsandlevers?
    What if agent state
    What if agent state
    complete sequence?
    ./output_images_with_boxes/image22.txt
    Mission complete, it took 0.441s
    
    Recognition Result:
    
    -The RL Probilem
    -State
    Fullvy ObservabeEnvironments
    Fullobservabilitvy:agent directly
    observes environment state
    acion
    S
    Agent state=environment
    information state
    state
    FormallythisisaMarkov
    decision process(MDP)
    (Nextlectureandthe
    majority ofthis course)
    ./output_images_with_boxes/image23.txt
    Mission complete, it took 0.674s
    
    Recognition Result:
    
    -TheRL Probiem
    —State
    Partially ObservableEnvironments
    Partial observabilitv:agent indirectly observes environment:
    Arobot with cameravision isnttold its absolutelocation
    A tradingagent onlv observes current prices
    Apoker playing agent only observes public cards
    Now agent state≠environment state
    Formally this isapartially observable Markov decision process
    (POMDP)
    Agent must construct its own state representation S,e.g.
    Complete history: S
    Beliefs of environment state: S= (P[Sg = s.mPS9 = s)
    Recurrent neural network: S=c(S +O.Yo)
    ./output_images_with_boxes/image24.txt
    Mission complete, it took 0.442s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Major Componentsof an RLAgent
    iAnRLagent may includeoneor more ofthesecomponemts:
    Policy:agent's behaviour function
    valuefunction:howgoodis each state and/or action
    Model:agent s representation oftheenvironment
    ./output_images_with_boxes/image25.txt
    Mission complete, it took 0.398s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Policy
    policy
    theagentsbehaviour
    ■tisamapfrom stateto action,e.g.
    Deterministic policy:a=r(s)
    Stochastic policy; 7(ajs) = P[A= a.S:= s]
    ./output_images_with_boxes/image26.txt
    Mission complete, it took 0.427s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    ValueFunction
    valuefunction is aprediction of futurereward
    Usedtoevaluatethegoodness/badness of states
    Andthereforetoselect between actions,e.g.
    (s)=BR1+7R2+?R3+…IS= s
    ./output_images_with_boxes/image27.txt
    Mission complete, it took 0.196s
    
    Recognition Result:
    
    Introdu
    —nside An RL Agent
    Example:ValueFunction inAtari
    ./output_images_with_boxes/image28.txt
    Mission complete, it took 0.345s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Mode
    model predicts what theenvironment willdonext
    predicts the next state
    尼predicts the next(immediate)reward,e.g.
    P=P[S1=sIS=s, =a]
    g=E[R1|S=s,=a]
    ./output_images_with_boxes/image29.txt
    Mission complete, it took 0.294s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Maze Exampe
    Start
    Rewards:
    per time-step
    Actions:N,ES.W
    States: Agent s location
    Goal
    ./output_images_with_boxes/image3.txt
    Mission complete, it took 0.389s
    
    Recognition Result:
    
    -Admi
    Assessment
    Assessment will be50% coursework,50exam
    Coursework
    Assignment A:RL problem
    Assignment B:Kernels problem
    Assessment 二 max(assznmentL.assiznmemt2)
    Examination
    A:3RLquestions
    B:3kernels questions
    Answer any 3questions
    ./output_images_with_boxes/image30.txt
    Mission complete, it took 0.279s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Maze Example:Policy
    Start
    Goal
    Arrows represent policy r(s)for each state
    ./output_images_with_boxes/image31.txt
    Mission complete, it took 0.286s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Maze Example:Value Function
    -13-12-11-10
    Start  -16 -15
    -22 -21
    Goal
    Numbers represent value -(s) of each state
    ./output_images_with_boxes/image32.txt
    Mission complete, it took 0.620s
    
    Recognition Result:
    
    Introdu
    -Inside An RL Agent
    Maze Example:Model
    Agent may have an internal
    modeloftheenvironment
    Start
    Dvnamics: how actions
    changethe state
    Rewards:howmuchreward
    fromeach state
    Goal
    Themodelmaybeimperfect
    Grid layoutrepresents transition model P,
    Numbers represent immediate rewardR from each state
    (samefor alla)
    ./output_images_with_boxes/image33.txt
    Mission complete, it took 0.474s
    
    Recognition Result:
    
    Introdu
    —nside An RL Agent
    Categorizing RL agents(1)
    value Based
    NoPolicv(Imolicit)
    ValueFunction
    Policy Based
    Policy
    ValueFunction
    Actor Critic
    Policy
    valueFunction
    ./output_images_with_boxes/image34.txt
    Mission complete, it took 0.289s
    
    Recognition Result:
    
    Introdu
    —nside An RL Agent
    Categorizing RL agents(2)
    Model Free
    Policy and/orvalue Function
    Mod
    ModelBased
    Policy and/orvalueFunction
    Model
    ./output_images_with_boxes/image35.txt
    Mission complete, it took 0.355s
    
    Recognition Result:
    
    Introdu
    Clside An RL Agent
    RLAgent Taxonomy
    Model-Free
    Actor
    Policv
    valueFunction
    一Critic
    Policv-Based
    value-Based
    Model-Based
    Model
    ./output_images_with_boxes/image36.txt
    Mission complete, it took 0.594s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Learning and Panning
    Twofundamentalproblems in sequentialdecisiom making
    Reinforcement Learning:
    The environment is initiallv unknown
    The agent interacts withtheenvironment
    The agent improves itspolicy
    Panning:
    model of the environment isknown
    Theagent performs computations with its model (without any
    externalinteraction)
    The agent improves its policy
    a.k.a.deliberation,reasoning,introspection,pondering,
    thought,search
    ./output_images_with_boxes/image37.txt
    Mission complete, it took 0.390s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Atari Example:ReinforcementLearning
    bservatior
    Rules of thegame are
    unknown
    Learn directfrom
    reward
    interactive game-play
    Pickactionson
    joystick,seepixels
    andscores
    ./output_images_with_boxes/image38.txt
    Mission complete, it took 0.405s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Atari Example:Planning
    Rules ofthegameareknown
    Can query emulato
    perfect model inside agent's brain
    Ifltake actionafrom state
    what would the next state be?
    what wouldthescorebe?
    Plan ahead tofind optimal policy
    e.g.tree search
    ./output_images_with_boxes/image39.txt
    Mission complete, it took 0.441s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Exploration andExploitation(I)
    Reinforcememnt learning isliketria-and-errorlearning
    The agent should discover agood policy
    Fromits experiences oftheenvironment
    Without losingtoo muchreward along the way
    ./output_images_with_boxes/image4.txt
    Mission complete, it took 0.627s
    
    Recognition Result:
    
    -Admi
    Textbooks
    An IntroductiontoReinforcementLearning,Suttonand
    Barto,1998
    MITPress.1998
    、40pounds
    Availabefree online!
    http://webdocs.cs.ualberta.ca/～sutton/book/the-book.html
    Algorithms for Reinforcement Learning,Szepesvari
    Morgan and Claypoo,2010
    20pound
    Availablefree online!
    http://www.ualberta.ca/～szepesva/ papers/RLAlgsInMDPs.pdf
    ./output_images_with_boxes/image40.txt
    Mission complete, it took 0.334s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    ExplorationandExploitation(2)
    Exporationfinds moreinformationaboutthe environment
    Expoitatiom exploitsknown informationto maximise reward
    ■tis usually important to explore as well as exploit
    ./output_images_with_boxes/image41.txt
    Mission complete, it took 0.550s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Examples
    Restaurant Selection
    ExploitationGotoyourfavouriterestaurant
    ExplorationTrv anew restaurant
    Online Banner Advertisements
    Exploitation Showthe most successful advert
    Exploration Show adifferent advert
    Oil Drilling
    Exploitation Drill at the best knownlocation
    Exploration Drill at anew location
    Game Playing
    Exploitation Playthe moveyoubelieve is best
    Exploration Play an experimentalmove
    ./output_images_with_boxes/image42.txt
    Mission complete, it took 0.278s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Prediction andControl
    iPrediction:evaluatethefuture
    Given apolicv
    Control:optimisethefuture
    Findthebest policy
    ./output_images_with_boxes/image43.txt
    Mission complete, it took 0.353s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    GridwordExampe:Prediction
    3.8 88445.31.5
    1.53.02.3190.5
    0.10.707040.4
    -1.0-04-04-0.6-1.2
    Actions
    1.9-1.3-1.2-1.4-20
    (b)
    (a)
    What isthevaluefunctionfor theuniformrandompolicy?
    ./output_images_with_boxes/image44.txt
    Mission complete, it took 0.393s
    
    Recognition Result:
    
    Introdu
    -Problems within RL
    Gridworld Example:Control
    22024422019.417.5
    19.822019.817816.0
    |17819.8178160144
    16.017816014413.0
    14.4]16.014413.011.7
    a)gridworld
    b)0
    C)T
    What istheoptimalvaluefunction over allpossible policies?
    What is the optimal policy?
    ./output_images_with_boxes/image45.txt
    Mission complete, it took 0.644s
    
    Recognition Result:
    
    -Course Outline
    Course Outline
    Part
    Elementarv ReinforcementLearning
    IntroductiontoRL
    Markov Decision Processes
    Planning bv Dynamic Programming
    Mode-Free Prediction
    可Mode-FreeControl
    Part lReinforcement Learning in Practice
    ValueFunction Approximation
    Policy Gradient Methods
    IntegratingLearning and Planning
    Exploration andExploitation
    Case study -RLin games
    ./output_images_with_boxes/image5.txt
    Mission complete, it took 0.495s
    
    Recognition Result:
    
    -AboutRL
    ManvFaces of Reinforcement Learning
    Computer Science
    Neuroe
    Engineering
    lMachine
    ewarc
    Optimal<
    Svstem
    ontrol
    Operations
    esearch
    onditionina
    Eounded
    Mathematics
    Psychology
    ationality
    Economics
    ./output_images_with_boxes/image6.txt
    Mission complete, it took 0.370s
    
    Recognition Result:
    
    -AboutRL
    Branches of MachineLearning
    Supervisee
    Unsupervised
    Learninq
    arnine
    Machine
    Learninq
    Feinforcement
    Learning
    ./output_images_with_boxes/image7.txt
    Mission complete, it took 0.517s
    
    Recognition Result:
    
    -AboutRL
    Characteristics of Reinforcement Learning
    What makes reinforcement learningdifferent from other machine
    learning paradigms?
    There isnosupervisor,only areward signal
    Feedback is delavyed,not instantaneous
    Time really matters(sequential,non ii.d data)
    Agent sactions affectthe subseauent datait receives
    ./output_images_with_boxes/image8.txt
    Mission complete, it took 0.388s
    
    Recognition Result:
    
    -AboutRL
    Examples of Reinforcement Learning
    helicopter
    Fly stunt manoeuvres in
    Defeat the world champion at Backgammon
    Manage an investment portfolio
    Controlapower station
    Make
    humanoidrobot walk
    Play many different Atarigames better than humans
    ./output_images_with_boxes/image9.txt
    Mission complete, it took 0.175s
    
    Recognition Result:
    
    -AboutRL
    Helicopter Manoeuvres


## 4. Showcase example outputs



```python
import cv2 as cv

output_dir = pathlib.Path("/home/lucassoares/Desktop/projects/mediumPosts/ocr_productivity/output_images_with_boxes")

# image = cv.imread(str(np.random.choice(list(output_dir.iterdir()),1)[0]))
image = cv.imread("/home/lucassoares/Desktop/projects/mediumPosts/ocr_productivity/output_images_with_boxes/image7.png")
size_reshaped = (int(image.shape[1]),int(image.shape[0]))

image = cv.resize(image, size_reshaped)
cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()
```

![image-2.png](Faster%20Notes%20with%20Python_files/markdown_19_attachment_0_0.png)




Text recognition sample:



```python
filename = "/home/lucassoares/Desktop/projects/mediumPosts/ocr_productivity/output_images_with_boxes/image7.txt"
with open(filename, "r") as text:
    for line in text.readlines():
        print(line.strip("\n"))
```

    -AboutRL
    Characteristics of Reinforcement Learning
    What makes reinforcement learningdifferent from other machine
    learning paradigms?
    There isnosupervisor,only areward signal
    Feedback is delavyed,not instantaneous
    Time really matters(sequential,non ii.d data)
    Agent sactions affectthe subseauent datait receives


### No more copy and paste 

What I like about this approach is that in the long run you can end up with a really powerful tool to transcribe all sorts of documents, from handwritten notes to random photos with text and more. For me it was an interesting way to explore a little bit of OCR as well as set up a pipeline for quickly going through a lecture with pdf slides.

---

If you liked this post connect with me on [Twitter](https://twitter.com/LucasEnkrateia), [LinkedIn](https://www.linkedin.com/in/lucas-soares-969044167/) and follow me on [Medium](https://lucas-soares.medium.com). 
Thanks and see you next time! :)
