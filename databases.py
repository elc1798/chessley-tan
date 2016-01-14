import pymongo

##Run to create databases and collections
connection = MongoClient()
db = connection['database']
db.createCollection('users')

## Adds a new user to the profile
def newUser(username, password):
    return 0
