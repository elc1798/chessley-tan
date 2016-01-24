from passlib.hash import pbkdf2_sha256
import pymongo
from pymongo import MongoClient

##Run to create databases and collections
connection = MongoClient()
db = connection.database
#db.createCollection('users')
#db.createCollection('bots')

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

## Adds a new user to the profile
def newUser(username, password):
    user = {"_id":str(numUsers()+1), 'un':username, 'pass':hashPass(password)}
    db.users.insert(user)
    return True

## Authenticates user
def authenticate(username, password):
    person = db.users.find({'un':username, 'pass':hashPass(password)})
    for p in person:
        return True
    return False

## Hashes and returns password that is hashed and salted by 29000 rounds of pbkdf2 encryption
def hashPass(password):
    return pbkdf2_sha256.encrypt(password)

## Returns true if password hashes into hashpass, false otherwise
def verify(password, hashpass):
    return pbkdf2_sha256.verify(password,hashpass)
