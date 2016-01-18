from passlib.hash import pbkdf2_sha256
import pymongo

##Run to create databases and collections
connection = MongoClient()
db = connection['database']
db.createCollection('users')
db.createCollection('bots')

def numUsers():
    profiles = db.users.find()
    x = 0
    for i in profiles:
        x += 1
    return x

## Adds a new user to the profile
def newUser(username, password):
    if len(db.users.find({'un':username})) != 0:
        return 1
    user = {"_id":str(numUsers()+1), "un":username, "pass":password}
    db.users.insert(user)
    return 0

## Hashes and returns password that is hashed and salted by 29000 rounds of pbkdf2 encryption
def hashPass(password):
    return pbkdf2_sha256.encrypt(password)

## Returns true if password hashes into hashpass, false otherwise
def verify(password, hashpass):
    return pbkdf2_sha256.verify(password,hashpass)
