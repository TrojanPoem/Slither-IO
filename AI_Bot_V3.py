from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import cv2
import numpy as np
from selenium.webdriver.common.by import By

import pyautogui

import random

#Global Control Variables
main_loop_active = True
Auto_respawn = True

#Global Constants
WND_HEIGHT = 512
WND_WIDTH = 256
WND_INIT_X = 0
WND_INIT_Y = 0
GAME_URL = 'http://www.slither.io'
HEADLESS_MODE = False
IMG_SAMPLE_TIME = 150 #Reaction time in millisecond
NICK_NAME = 'J3adHeart'
SNAKE_VISION_WND_X = 700
SNAKE_VISION_WND_Y = 300
MIN_X_COORD = 10
MAX_X_COORD = 560
MIN_Y_COORD = 110
MAX_Y_COORD = 620

#ID of HTML elements
NICKNAME_INPUT_ID = 'nick'
LASTSCORE_LABEL_ID = 'lastscore'

class client:
    
    #Initialize the bot client with specific dimensions and position, then navigate to the game URL.
    #headless_mode: if true, firefox browser window is hidden (not rendered).
    def __init__(self, URL, width, height, wnd_init_x, wnd_init_y, headless_mode):

        self.last_error = ''
        #Prepare options of firefox
        options = FirefoxOptions()
        
        if headless_mode:
           options.add_argument('-headless')

        #Create the driver
        self.driver = webdriver.Firefox(options=options)

        #Set the initial window positon and size
        self.driver.set_window_size(width, height)
        self.driver.set_window_position(wnd_init_x, wnd_init_y)

        #Navigate to the game URL.
        self.driver.get(URL)

    #Returns a screenshot of the client in gray scale with only edges.
    def screenshot(self, hyst_upper_th = 180, hyst_lower_th = 180):
        # Capture a screenshot of the webpage
        screenshot = self.driver.get_screenshot_as_png()

        # Convert the PNG image data to a NumPy array
        np_image = np.frombuffer(screenshot, np.uint8)

        # Decode the NumPy array to an OpenCV image format
        cv_image = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE)

        #Detect the edges using canny filter
        edge_img = cv2.Canny(cv_image, hyst_upper_th, hyst_lower_th)

        return edge_img

    #Sets the player name
    def set_nick_name(self, nick_name):
        
        #Find the input box by its id
        nick_name_input = self.driver.find_element(By.ID, NICKNAME_INPUT_ID)

        #clear the field
        nick_name_input.clear()

        #Send the text to it
        nick_name_input.send_keys(nick_name)

    #Get the last game score
    def get_lastscore(self):

        try:
          
            last_score_lbl = self.driver.find_element(By.ID, LASTSCORE_LABEL_ID)
            score = int(last_score_lbl.text.split(' ')[-1])
            
        except Exception as e:
            
            self.last_error = e
            #When the game first loads, the element does not exist -> exception.
            score = 0   

        return score

    #Get the player rank
    def get_rank(self):

        try:
            
            rank_lbl = self.driver.find_element(By.XPATH, '/html/body/div[13]/span[3]');
            rank = int(rank_lbl.text)

        except Exception as e:
            
            self.last_error = e
            rank = 0
            
        return rank

    #Get the player rank
    def get_total_number_of_players(self):

        try:
            
            total_players_lbls = self.driver.find_element(By.XPATH, '/html/body/div[13]/span[5]');
            total_players = int(total_players_lbls.text)

        except Exception as e:
            
            self.last_error = e
            total_players = 0
            
        return total_players


    #Get the current score
    def get_current_score(self):

        try:
            
            cur_score_lbl = self.driver.find_element(By.XPATH, '/html/body/div[13]/span[1]/span[2]');
            cur_score = int(cur_score_lbl.text)

        except Exception as e:
            
            self.last_error = e
            cur_score = 0
            
        return cur_score


    #Print the game stats including: score, rank, total players, last score.
    #Update: DISPLAY THE status each 3 sec or so, and ability to modify the printed values instead of re-printing.
    def show_stats(self):
        
        current_score = self.get_current_score()
        total_players = self.get_total_number_of_players()
        rank = self.get_rank()
        last_score = self.get_lastscore()

        print('[+] Statstics:\n\tCurrent Score:{}\n\tTotal Players:{}\n\tRank:{}\n\tLast Score:{}\n\t'.format(current_score, total_players, rank, last_score))


    #Start the game by clicking the play button
    def start_game(self):

        try:
        
            #div as a button, find it then click. The button is the div with class name of "btnt nsi sadg1".
            btn = self.driver.find_element(By.XPATH, '/html/body/div[2]/div[5]/div')
            #btn click

            self.driver.execute_script("arguments[0].click();", btn)
            #FAILS if the object is obscured
            #btn.click()

        except Exception as e:
            
            self.last_error = e
            return False

        return True
    
    #Checks if the set your name and play button screen is shown (main screen).
    def is_main_screen(self):

        try:
            
            #I noticed that this object has an opacity of 1 when playing and 0 when not playing, so i decided to use it to find if we are in the main screen.
            unique_obj = self.driver.find_element(By.ID, "nskh")
            
            is_main_scrn = not bool(int(unique_obj.value_of_css_property('opacity')))

        except Exception as e:
            
            self.last_error = e  
            is_main_scrn = True

        return is_main_scrn

    def get_last_error(self):
        return self.last_error


    #UNFINISHED ............................................############################################ FUNCTION ####################  
    def remove_HUD(self):

        try:
            
            self.driver.execute_script("document.getElementsByClassName(""nsi"").style.display = 'none';")

        except Exception as e:
            
            self.last_error = e
            return False

        return True
    
    def uninit(self):
        self.driver.quit()

    #Move the mouse to a coordinate
    def moveTo(self, x, y):
        if x > MAX_X_COORD:
            self.last_error = 'X coordinate is more than the maximum.'
            return False
        
        if x < MIN_X_COORD:
            self.last_error = 'X coordinate is less than the minimum.'
            return False

        
            
        if y > MAX_Y_COORD:
            self.last_error = 'Y coordinate is more than the maximum.'
            return False
        
        if y < MIN_Y_COORD:
            self.last_error = 'Y coordinate is less than the minimum.'
            return False

        pyautogui.moveTo(x, y)

        return True
        
#A random player agent
class random_agent:
    def __init__(self, game):
        self.env = game
        
    def perform_action(self):

        x_target = random.uniform(MIN_X_COORD, MAX_X_COORD)
        y_target = random.uniform(MIN_Y_COORD, MAX_Y_COORD)
        
        if not self.env.moveTo(x_target, y_target):
            print('[+] Agent selected coordinates are outside the allowale boundary.\n\tError:{}'.format(self.env.get_last_error()))
            

class DDPG_agent:
    
    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])  
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        print("We finished building the model")
        return model, model.trainable_weights, S

    #In the TORCS there are 18 different types of sensor input,
    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')    
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)  
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("We finished building the model")
        return model, A, S

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def use(self):
        for j in range(max_steps):
            a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            ob, r_t, done, info = env.step(a_t[0])

        
#The main loop of the bot
def main():
    game = client(GAME_URL, WND_WIDTH, WND_HEIGHT, WND_INIT_X, WND_INIT_Y, HEADLESS_MODE)


    #Start the game the first time after choosing a name..... Fails
    if game.is_main_screen():

        game.set_nick_name(NICK_NAME)
        
        if not game.start_game():
            print('[+] Failed to click the game play button\n\tError:{}'.format(game.get_last_error()))

    rand_agent = random_agent(game)
    
    while main_loop_active:


        if game.is_main_screen():
            if not game.start_game():
                print('[+] Failed to click the game play button\n\tError:{}'.format(game.get_last_error()))
            else:
                print('[+] Restarting the game...')
        else:
            rand_agent.perform_action()
        
        frame = game.screenshot()

        '''
        Remove the game HUD
        if game.remove_HUD():
            print('[+]HUD was removed successfully.')
        else:
            print('[+] Failed to hide the HUD.\n\tError:{}'.format(game.get_last_error()))
        '''

        
        # Display the black and white image
        cv2.imshow("Snake Vision", frame)
        cv2.moveWindow("Snake Vision", SNAKE_VISION_WND_X, SNAKE_VISION_WND_Y)
        cv2.waitKey(IMG_SAMPLE_TIME)

    #Unload the drive
    game.uninit()


#Run the main procedure
main()

