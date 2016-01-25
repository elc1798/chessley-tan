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
def updateRanks():
    return True

# Updates the win/losses/draws of player and then changes their ELO, with a successive call to updateRanks()
# score is an integer of either 1(win), 0(draw), or -1(loss) and updates the database likewise, user is the username
def updateScore(score, user):
    # Stuff to update the score
    # Stuff to change ELO
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
    print hashPass
    return verify(password, hashPass)

# Hashes and returns password that is hashed and salted by 29000 rounds of pbkdf2 encryption
def hashPass(password):
    return pbkdf2_sha256.encrypt(password)

# Returns true if password hashes into hashpass, false otherwise
def verify(password, hashpass):
    return pbkdf2_sha256.verify(password,hashpass)
