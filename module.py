from passlib.hash import pbkdf2_sha256
import pymongo
from pymongo import MongoClient

connection = MongoClient()
db = connection.database

def numUsers():
    profiles = db.users.find()
    x = 0
    for i in profiles:
        x += 1
    return x

def userExists(username):
    person = db.users.find({'un':username})
    for p in person:
        return True
    return false

# Adds a new user to the profile
def newUser(username, password):
    user = {"_id":str(numUsers()+1), 'un':username, 'pass':hashPass(password), 'rank':str(numUsers()+1), 'elo':'0', 'wins':'0', 'losses':'0', 'draws':'0'}
    db.users.insert(user)
    return True

# Updates the ranks based upon ELO
# O(god) but I'm kinda tired rn --Jion
def updateRanks():
    for user1 in db.users:
        user1["rank"] = 0
        for user2 in db.users:
            if(user1["elo"]<user2["elo"]):
                user1["rank"]+=1
    return True

# Updates the win/losses/draws of player and then changes their ELO, with a successive call to updateRanks()
# score is an integer of either 1(win), 0(draw), or -1(loss) and updates the database likewise, user1 is the username
# user2 is the user that user1 played against
# This function should be called once per game and score is relative to user1
def updateScore(score, user1, user2):
    user1 = db.users[user1]
    user2 = db.users[user2]
    elo1 = user1["elo"]
    elo2 = user2["elo"]
    delta = int(elo1)+int(elo2)
    # Stuff to update the score
    if(score == 1):
        user1["wins"] = str(int(user1["wins"])+1)
        user2["losses"] = str(int(user1["losses"])+1)
        
        user1["elo"] += delta/2
        user2["elo"] -= delta/2
    elif(score == -1):
        user2["wins"] = str(int(user1["wins"])+1)
        user1["losses"] = str(int(user1["losses"])+1) 
        
        user2["elo"] += delta/2
        user1["elo"] -= delta/2
    else:
        user1["draws"] = str(int(user1["draws"])+1)
        user2["draws"] = str(int(user1["draws"])+1)
        
    #update ranks
    return updateRanks()

# Returns the dictionary of the player except for the password
def getUser(user):
    return db.users.find({'un':user}, {'pass':0})[0]

# Authenticates user
def authenticate(username, password):
    person = db.users.find({'un': username})
    try:
        hashPass = person[0]['pass']
    except:
        return False
    #print hashPass
    return verify(password, hashPass)

# Hashes and returns password that is hashed and salted by 29000 rounds of pbkdf2 encryption
def hashPass(password):
    return pbkdf2_sha256.encrypt(password)

# Returns true if password hashes into hashpass, false otherwise
def verify(password, hashpass):
    return pbkdf2_sha256.verify(password,hashpass)
