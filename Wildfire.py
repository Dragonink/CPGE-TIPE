'''
    WILDFIRE
    ========

    Classes
    -------
    Tree : Represents a single tree with its phyiscal properties.
    Forest : Represents a forest containing several trees in a matrix.
    Wildfire : Contains all the states the forest was in at different time frames.

    How to use
    ----------
    (1) Create a Forest object.
        "forest = Forest(<dim>, <start>, [wind])"
        <dim> : Dimension of the forest
        <start> : Array containing the coordinates of the starting points, or "Forest.rndStart(<n>)" to compute n random coordinates.
        [wind] : Wind direction [-1 ~ 7]

    (2) Create a Wildfire object with the previous Forest object.
        "wildfire = Wildfire(forest, <model>, [dt])"
        <model> : Function returning a new Forest object given another one
        [dt] : Factor for the time elapsed between two frames

    (3) Compute new frames of the wildfire.
        "wildfire(<frames>)"
        <frames> : Number of frames to compute

    (4) Display the results.
        "wildfire.show([frames...])" | "wildfire.details()" | "wildfire.anim([fps])"
        [frames...] : Frames to display
        [fps] : Frames per second displayed
'''


import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from math import ceil
from random import random, randint
from time import time
from types import LambdaType


### CLASSES
class Tree :
    '''This represents a single tree.

        Arguments
        ----------
        state : State self is in (0 = normal, 1 = burning, 2 = burnt).
        temperature : Temperature of self [K].
        grass : Height of the grass near the tree [m].
        forest : Forest object self belongs to.

        Properties
        ----------
        state : State self is in (0 = normal, 1 = burning, 2 = burnt).
        temperature : Temperature of self [K].
        grass : Height of the grass near the tree [m].
        Forest : Forest object self belongs to.
        location : Tuple containing the coordinates of self in Forest.
    '''
    def __init__(self, state:int=0, temperature:float=298.15,  grass:float=None, humidity:float=None, forest:'Forest'=None) :
        self.state = state
        self.temperature = temperature
        self.grass = 0.48#grass if not grass is None and grass >= 0 else random() * 0.5
        self.humidity = 0.2#humidity if not humidity is None and 0 <= humidity <= 1 else random()
        if (not forest is None) and isinstance(forest[0], Forest) :
            self.Forest = forest[0]
            self.location = forest[1], forest[2]
        else :
            self.Forest, self.location = None, None

        self._neighbours = None
    def __repr__(self) :
        return "Tree({0},{1},{2},{3})".format(self.state, self.temperature, self.grass, self.humidity)
    ## Computed properties
    @property
    def neighbours(self) -> list :
        '''Computes and sets the "neighbours" property.

            Returns
            -------
            A list containing the Moore neighbourhood (N, NE, E, SE, S, SW, W, NW) of self,
            in which an item is either a Tree object or None if the neighbour is not in the forest.

            Raises
            ------
            ValueError if self is not in a Forest object.
        '''
        if self.Forest is None :
            raise ValueError('self is not in a Forest')
        if self._neighbours is None :
            i, j = self.location
            self._neighbours = []
            for c in [[i-1,j],[i-1,j+1],[i,j+1],[i+1,j+1],[i+1,j],[i+1,j-1],[i,j-1],[i-1,j-1]] : # N,NE,E,SE,S,SW,W,NW
                if 0 <= c[0] < self.Forest.dim and 0 <= c[1] < self.Forest.dim :
                    self._neighbours.append(self.Forest[c[0],c[1]])
                else :
                    self._neighbours.append(None)
        return self._neighbours
    ## Utilities
    def windDirCalc(self, origin:tuple) -> float :
        '''Computes the cosinus of the angle between self and the wind axis of the forest.

            Arguments
            ---------
            origin : Tuple containing the coordinates of a burning tree.

            Returns
            -------
            A float equal to the cosine of the angle between the wind axis and self.

            Raises
            ------
            ValueError if self is not in a Forest object.
        '''
        if self.Forest is None :
            raise ValueError('self is not in a Forest')
        if self.Forest.wind == -1 : # Wind : null
            return 0
        tree = (self.location[0] - origin[0], self.location[1] - origin[1]) # Coordinates of self relative to origin
        if tree == (0,0) : # self is origin
            return 0
        wind = (0,0)
        if self.Forest.wind == 0 : # Wind : N
            wind = (-1,0)
        elif self.Forest.wind == 1 : # Wind : NE
            wind = (-1,1)
        elif self.Forest.wind == 2 : # Wind : E
            wind = (0,1)
        elif self.Forest.wind == 3 : # Wind : SE
            wind = (1,1)
        elif self.Forest.wind == 4 : # Wind : S
            wind = (1,0)
        elif self.Forest.wind == 5 : # Wind : SW
            wind = (1,-1)
        elif self.Forest.wind == 6 : # Wind : W
            wind = (0,-1)
        else : # Wind : NW
            wind = (-1,-1)
        return (tree[0] * wind[0] + tree[1] * wind[1]) / (((tree[0] ** 2 + tree[1] ** 2) * (wind[0] ** 2 + wind[1] ** 2))  ** .5) # Dot product / Norms product
    @staticmethod
    def copy(self, forest:'Forest'=None) -> 'Tree' :
        '''Creates a copy of self.

            Arguments
            ---------
            forest : Forest object self belongs to.

            Returns
            -------
            A new Tree object with the same properties as self.
        '''
        if forest is None :
            return Tree(state=self.state, temperature=self.temperature, grass=self.grass, humidity=self.humidity)
        else :
            return Tree(state=self.state, temperature=self.temperature, grass=self.grass, humidity=self.humidity, forest=[forest, self.location[0], self.location[1]])

class Forest :
    '''This represents a forest containing several trees.
        This can be used as an iterator to loop over all the Tree objects.

        Arguments
        ---------
        dim : Dimension of the forest.
        start : List of tuples containing the coordinates of the trees to burn. Can be "Forest.rndStart(<n>)" to compute n random coordinates.
        wind : Direction of the wind (-1 = No wind, 0 = N, 1 = NE, 2 = E, 3 = SE, 4 = S, 5 = SW, 6 = W, 7 = NW).

        Properties
        ----------
        __Forest : Matrix containing the trees.
        dim : Dimension of the forest.
        wind : Direction of the wind (-1 = No wind, 0 = N, 1 = NE, 2 = E, 3 = SE, 4 = S, 5 = SW, 6 = W, 7 = NW).
        start : List of tuples containing the coordinates of the first trees to burn.
    '''
    def __init__(self, dim:int, start:list or LambdaType=[], wind:int=-1) :
        self.__Forest = np.array([[Tree(forest=[self,i,j]) for j in range(dim)] for i in range(dim)])
        self.dim = dim
        self.wind = wind
        self.start = []
        if isinstance(start, LambdaType) :
            self.start = start(dim)
        elif isinstance(start, list) :
            self.start = start
        for (i,j) in [(i,j) for (i,j) in self.start if 0 <= i < dim and 0 <= j < dim] :
            self.__Forest[i,j].state = 1
            self.__Forest[i,j].temperature = 573.15
            self.__Forest[i,j].humidity = 0

        self._Wind = None
        self._State = None
        self._Temperature = None
        self._Grass = None
        self._Humidity = None
    def __repr__(self) :
        return "Forest({0},{1},{2}) :\n".format(self.dim, self.start, self.Wind) + str(self.__Forest)
    ## Computed properties
    @property
    def Wind(self) -> str :
        '''Computes and sets the "Wind" property.

            Returns
            -------
            A string representing the wind direction.
        '''
        if self._Wind is None :
            self._Wind = ''
            if -1 < self.wind <= 1 or self.wind == 7 :
                self._Wind += 'N'
            elif 3 <= self.wind <= 5 :
                self._Wind += 'S'
            if 1 <= self.wind <= 3 :
                self._Wind += 'E'
            elif 5 <= self.wind <= 7 :
                self._Wind += 'W'
            self._Wind = '_' if self._Wind == '' else self._Wind
        return self._Wind
    @property
    def State(self) -> np.ndarray :
        '''Computes and sets the "State" property.

            Returns
            -------
            A matrix containing the "state" property of the Tree objects.
        '''
        if self._State is None :
            self._State = np.vectorize(lambda T : T.state)(self.__Forest)
        return self._State
    @property
    def Temperature(self) -> np.ndarray :
        '''Computes and sets the "Temperature" property.

            Returns
            -------
            A matrix containing the "temperature" property of the Tree objects.
        '''
        if self._Temperature is None :
            self._Temperature = np.vectorize(lambda T : T.temperature)(self.__Forest)
        return self._Temperature
    @property
    def Grass(self) -> np.ndarray :
        '''Computes and sets the "Grass" property.

            Returns
            -------
            A matrix containing the "grass" property of the Tree objects.
        '''
        if self._Grass is None :
            self._Grass = np.vectorize(lambda T : T.grass)(self.__Forest)
        return self._Grass
    @property
    def Humidity(self) -> np.ndarray :
        '''Computes and sets the "Humidity" property.

            Returns
            -------
            A matrix containing the "humidity" property of the Tree objects.
        '''
        if self._Humidity is None :
            self._Humidity = np.vectorize(lambda T : T.humidity)(self.__Forest)
        return self._Humidity
    ## Iterator protocol
    def __len__(self) :
        return self.dim ** 2
    def __getitem__(self, key) :
        return self.__Forest[key]
    def __iter__(self) :
        self.idx = 0
        forest = []
        for r in self.__Forest :
            forest += [T for T in r]
        return iter(forest)
    def __next__(self) :
        if self.idx < len(self) :
            idx, self.idx = self.idx, self.idx + 1
            return idx
        else :
            del self.idx
            raise StopIteration
    ## Utilities
    @staticmethod
    def rndStart(n:int=1) -> '(int) -> list' :
        '''Computes an array of random coordinates.
            To be used only as the "start" argument while constructing a Forest instance.

            Arguments
            ---------
            n : Number of start points to compute.

            Returns
            -------
            A lambda function to compute the random coordinates.
        '''
        def f(n, dim) :
            start = []
            while len(start) < n :
                i, j = randint(0, dim), randint(0, dim)
                if not (i,j) in start :
                    start.append((i,j))
            return start
        return lambda dim : f(n, dim)
    @staticmethod
    def distCalc(a:tuple, b:tuple) -> float :
        '''Computes the distance between two trees given their coordinates.

            Arguments
            ---------
            a : Tuple of the coordinates of the first tree.
            b : Tuple of the coordinates of the second tree.

            Returns
            -------
            A float equal to the distance [m] between two trees.
        '''
        return 3 * (((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** .5)
    @staticmethod
    def copy(self) -> 'Forest' :
        '''Creates a copy of self.

            Returns
            -------
            A new Forest object with the same properties as self.
        '''
        cp = Forest(self.dim, self.start, self.wind)
        for i in range(cp.dim) :
            for j in range(cp.dim) :
                cp.__Forest[i,j] = Tree.copy(self[i,j], forest=cp)
        return cp

class Wildfire :
    '''This represents an history list of the states the forest was in at different time frames.
        This can be used as an iterator to loop over all the states of the forest.
        This can be called to compute new states of the forest.

        Arguments
        ---------
        forest : Forest object self will start from.
        model : Function returning a new Forest object given a Wildfire object.
        dt : Factor for the time elapsed between two frames.

        Properties
        ----------
        __History : List containing the states of the Forest.
        model : Function returning a new Forest object given a Wildfire object.
        dt : Factor for the time elapsed between two frames.
        physics : Dictionary containing several values.
    '''
    def __init__(self, forest:'Forest', model:'(Wildfire) -> Forest', dt:float=1) :
        if not isinstance(forest, Forest) :
            raise TypeEror('forest is not a Forest instance')
        if len(forest.start) == 0 :
            raise ValueError('forest has no start tree')
        self.__History = [forest]
        self.model = model
        self.dt = dt

        self._physics = None
    def __repr__(self) :
        repr = ""
        for t, F in enumerate(self) :
            repr += "[{0}] {1}".format(t, F) + "\n"
        return repr
    ## Computed properties
    @property
    def physics(self) -> dict :
        '''Computes and sets the "physics" property.

            Returns
            -------
            A dictionary containing the following values :
                v0 : Function returning the speed of the flames without wind given the height of the grass.
                v1 : Function returning the speed of the flames with wind given the height of the grass.
                Time : Matrix containing the time before the tree will start to burn.
                flameOrigin : Matrix containing the index of the start point the flames will come from.
        '''
        if self._physics is None :
            # Compute the speeds of the flames without and with wind
            v0 = lambda d,h: (5.670e-8 * 1300.15 ** 4 * d) / (2 * 0.2 * (1200 * (573.15 - 298.15) + 2256e3 * h))
            v1 = lambda d,h: v0(d,h) * 0.90 * 2 * (1 + np.sin(0.35) - np.cos(0.35)) / d
            # Compute the time before the trees will burn and where the flames will come from
            Time = np.array([[0 for j in range(self[0].dim)] for i in range(self[0].dim)])
            flameOrigin = np.array([[-1 for j in range(self[0].dim)] for i in range(self[0].dim)])
            for i in range(self[0].dim) :
                for j in range(self[0].dim) :
                    if (i,j) in self[0].start :
                        Time[i,j] = 0
                        flameOrigin[i,j] = self[0].start.index((i,j))
                    else :
                        min, smin = float('inf'), -1
                        for (si,sj) in self[0].start :
                            f = None
                            if self[0].wind == -1 :
                                f = lambda t: Forest.distCalc((i,j), (si,sj)) / v0(self[0][int(t*i+(1-t)*si), int(t*j+(1-t)*sj)].grass, self[0][int(t*i+(1-t)*si), int(t*j+(1-t)*sj)].humidity) # Distance between the tree and the start point #s / Speed without wind
                            else :
                                f = lambda t: Forest.distCalc((i,j), (si,sj)) / ((1.3 + self[0][i,j].windDirCalc((si,sj))) * v1(self[0][int(t*i+(1-t)*si), int(t*j+(1-t)*sj)].grass, self[0][int(t*i+(1-t)*si), int(t*j+(1-t)*sj)].humidity)) # Distance between the tree and the start point #s / (1.3 + Cosine of angle) * Speed with wind
                            t = quad(f,0,1)[0]
                            if t < min :
                                min = t
                                smin = self[0].start.index((si,sj))
                        Time[i,j] = ceil(min)
                        flameOrigin[i,j] = smin
            self._physics = {
                'v0': v0,
                'v1': v1,
                'Time': Time,
                'flameOrigin': flameOrigin
            }
        return self._physics
    ## Iterator protocol
    def __len__(self) :
        return len(self.__History)
    def __getitem__(self, key) :
        return self.__History[key]
    def __iter__(self) :
        self.idx = 0
        return iter(self.__History)
    def __next__(self) :
        if self.idx < len(self.__History) :
            idx, self.idx = self.idx, self.idx + 1
            return idx
        else :
            del self.idx
            raise StopIteration
    ## Computation of new frames
    def __call__(self, iter:int) -> None :
        '''Computes new states of the forest.

            Arguments
            ---------
            iter : Number of new states to compute.
        '''
        t0 = time()
        t = t0
        for k in range(iter) :
            self.__History.append(self.model(self))
            if time() - t >= 5 :
                print("PROGRESS : {0} % | {1}/{2} frames".format(round(100*(k+1)/iter, 1), k+1, iter))
                t = time()
        print("COMPLETE : {0}s".format(round(time()-t0, 2)))
    ## Graphs
    def details(self, frame:int=None, a:tuple=None, b:tuple=None) -> None :
        '''Displays details of the Forest.

            Arguments
            ---------
            A : Tuple of coordinates
            B : Tuple of coordinates
        '''
        frame = frame if not frame is None else len(self) - 1
        fig = plt.figure(num="Details of Forest({0},{1},{2})".format(self[0].dim, self[0].start, self[0].Wind))
        if a is None or b is None or a == b :
            # Grass
            axGrass = plt.subplot(141)
            axGrass.set_axis_off()
            grass = axGrass.matshow(self[0].Grass, cmap='Greens', aspect='equal')
            cbGrass = plt.colorbar(grass, ax=axGrass, cmap='Greens', orientation='horizontal')
            cbGrass.set_label('Grass height [m]', weight='bold')
            cbGrass.set_clim(0)
            # Humidity
            axHumidity = plt.subplot(142)
            axHumidity.set_axis_off()
            humidity = axHumidity.matshow(self[0].Humidity, cmap='Blues', aspect='equal')
            cbHumidity = plt.colorbar(humidity, ax=axHumidity, cmap='Blues', orientation='horizontal')
            cbHumidity.set_label('Grass humidity [%]', weight='bold')
            cbHumidity.set_clim(0,1)
            # Flame origin
            axOrigin = plt.subplot(143)
            axOrigin.set_axis_off()
            origin = axOrigin.matshow(self.physics['flameOrigin'], cmap='rainbow', aspect='equal')
            t = [s for s in range(len(self[0].start))]
            cbOrigin = plt.colorbar(origin, ax=axOrigin, cmap='rainbow', ticks=t, orientation='horizontal')
            cbOrigin.ax.set_xticklabels([coords for coords in self[0].start])
            cbOrigin.set_label('Flame origin', weight='bold')
            # Time
            axTime = plt.subplot(144)
            axTime.set_axis_off()
            time = axTime.matshow(self.physics['Time'], cmap='Spectral', aspect='equal')
            cbTime = plt.colorbar(time, ax=axTime, cmap='Spectral', orientation='horizontal')
            cbTime.set_label('Time [s]', weight='bold')
        else :
            (ai,aj),(bi,bj) = a,b
            T = np.linspace(0,1,100)
            # Grass
            Grass = np.vectorize(lambda t: self[frame][int(t*ai+(1-t)*bi),int(t*aj+(1-t)*bj)].grass)(T)
            axGrass = plt.subplot(131)
            axGrass.set_title('Grass height profile [m]', fontweight='bold')
            axGrass.set_xlim(0,1)
            axGrass.set_ylim(0)
            grass = axGrass.plot(T, Grass, 'g')
            axGrass.fill_between(T, Grass, 0, color='g')
            # Humidity
            Humidity = np.vectorize(lambda t: self[frame][int(t*ai+(1-t)*bi),int(t*aj+(1-t)*bj)].humidity)(T)
            axHumidity = plt.subplot(132)
            axHumidity.set_title('Grass humidity profile [%]', fontweight='bold')
            axHumidity.set_xlim(0,1)
            axHumidity.set_ylim(0,1)
            humidity = axHumidity.plot(T, Humidity, 'b')
            axHumidity.fill_between(T, Humidity, 0, color='b')
            # Speeds
            Speed0 = np.vectorize(lambda t: self.physics['v0'](self[frame][int(t*ai+(1-t)*bi), int(t*aj+(1-t)*bj)].grass,self[frame][int(t*ai+(1-t)*bi), int(t*aj+(1-t)*bj)].humidity))(T)
            Speed1 = np.vectorize(lambda t: self.physics['v1'](self[frame][int(t*ai+(1-t)*bi), int(t*aj+(1-t)*bj)].grass,self[frame][int(t*ai+(1-t)*bi), int(t*aj+(1-t)*bj)].humidity))(T)
            axSpeeds = plt.subplot(133)
            axSpeeds.set_title('Flames speed [m/s]', fontweight='bold')
            axSpeeds.set_xlim(0,1)
            axSpeeds.set_ylim(0)
            speed0 = axSpeeds.plot(T, Speed0, 'y', label='$v_{0}$')
            speed1 = axSpeeds.plot(T, Speed1, 'r', label='$v_{1}$')
            axSpeeds.legend(loc='upper right')

        plt.show()
    def show(self, *frames:tuple) -> None :
        '''Displays one or several states Forest states.

            Arguments
            ---------
            frames : One or several states to be displayed.
        '''
        frames = frames if len(frames) > 0 else [len(self) - 1]
        fig = plt.figure(num='{0} Forest({1},{2})'.format(frames, self[frames[0]].dim, self[frames[0]].Wind))
        for k, frame in enumerate(frames) :
            # State
            axState = plt.subplot(len(frames), 2, k * 2 + 1)
            axState.set_axis_off()
            state = axState.matshow(self[frame].State, cmap=ListedColormap(['green','red','grey']), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), aspect='equal')
            cbState = plt.colorbar(state, ax=axState, cmap=ListedColormap(['green','red','grey']), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), ticks=[0,1,2], orientation='horizontal')
            cbState.ax.set_xticklabels(['Normal', 'Burning', 'Burnt'])
            # Temperature
            axTemperature = plt.subplot(len(frames), 2, k * 2 + 2)
            axTemperature.set_axis_off()
            temperature = axTemperature.matshow(self[frame].Temperature, cmap='jet', aspect='equal')
            cbTemperature = plt.colorbar(temperature, ax=axTemperature, cmap='jet', orientation='horizontal')
            cbTemperature.set_clim(np.min(self[0].Temperature), np.max(self[-1].Temperature))

            if k == len(frames) - 1 :
                cbState.set_label('State', weight='bold')
                cbTemperature.set_label('Temperature [K]', weight='bold')
        plt.show()
    def anim(self, fps:int=5) -> None :
        '''Shows an animation of the Wildfire.

            Arguments
            ---------
            fps : Number of frames per second displayed.
        '''
        fig = plt.figure(num='[0,{0}] Forest({1},{2})'.format(len(self) - 1, self[0].dim, self[0].Wind))
        # State
        axState = fig.add_subplot(121)
        axState.set_axis_off()
        state = axState.matshow(self[0].State, cmap=ListedColormap(['green','red','grey']), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), aspect='equal')
        cbState = fig.colorbar(state, ax=axState, cmap=ListedColormap(['green','red','grey']), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), ticks=[0,1,2], orientation='horizontal')
        cbState.set_label('State', weight='bold')
        cbState.ax.set_xticklabels(['Normal', 'Burning', 'Burnt'])
        # Temperature
        axTemperature = fig.add_subplot(122)
        axTemperature.set_axis_off()
        temperature = axTemperature.matshow(self[0].Temperature, cmap='jet', aspect='equal')
        temperature.set_clim(np.min(self[0].Temperature), np.max(self[-1].Temperature))
        cbTemperature = fig.colorbar(temperature, ax=axTemperature, cmap='jet', orientation='horizontal')
        cbTemperature.set_label('Temperature [K]', weight='bold')
        cbTemperature.set_clim(np.min(self[0].Temperature), np.max(self[-1].Temperature))

        def animate(t) :
            state = axState.matshow(self[t].State, cmap=ListedColormap(['green','red','grey']), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), aspect='equal')
            temperature = axTemperature.matshow(self[t].Temperature, cmap='jet', aspect='equal')
            temperature.set_clim(np.min(self[0].Temperature), np.max(self[-1].Temperature))
            return [state, temperature]
        ani = FuncAnimation(fig, animate, frames=len(self), interval=(1000/fps), blit=True)
        plt.show()


### MODELS
def Model1(wildfire:'Wildfire') -> 'Forest' :
    '''(1) EMPIRICAL MODEL.

        Arguments
        ---------
        wildfire : Wildfire object to compute the new state from.

        Returns
        -------
        A new Forest object corresponding to the new frame of the Wildfire.
    '''
    nextForest = Forest.copy(wildfire[-1])
    for T in nextForest :
        if T.state == 1 : # State : Burning
            T.temperature += wildfire.dt * 25
            for N in [N for N in T.neighbours if not N is None] :
                N.temperature += wildfire.dt * 15 * (1 + N.windDirCalc(T.location))
        elif T.state == 2 : # State : Burnt
            avg, n = 0, 0
            for N in [N for N in T.neighbours if not N is None] :
                avg += wildfire[-1][N.location].temperature
                n += 1
            T.temperature = avg / n
    for T in nextForest :
        if T.state == 0 and T.temperature >= 573.15 :
            T.state = 1
        elif T.state == 1 and T.temperature >= 5573.15 :
            T.state = 2
    return nextForest

def Model2(wildfire:'Wildfire') -> 'Forest' :
    '''(2) SEMI-PHYSICAL MODEL after Moretti's formula.

        Arguments
        ---------
        wildfire : Wildfire object to compute the new state from.

        Returns
        -------
        A new Forest object corresponding to the new frame of the Wildfire.
    '''
    nextForest = Forest.copy(wildfire[-1])
    for T in nextForest :
        if T.state == 0 : # State : Normal
            T.temperature += wildfire.dt * (573.15 - wildfire[0][T.location].temperature) / wildfire.physics['Time'][T.location]
        elif T.state == 1 : # State : Burning
            T.temperature += wildfire.dt * 25
        else : # State : Burnt
            avg, n = 0, 0
            for N in [N for N in T.neighbours if not N is None] :
                avg += wildfire[-1][N.location].temperature
                n += 1
            T.temperature = avg / n
    for T in nextForest :
        if T.state == 0 and T.temperature >= 573.15 :
            T.state = 1
        elif T.state == 1 and T.temperature >= 5573.15 :
            T.state = 2
    return nextForest
