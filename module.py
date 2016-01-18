from passlib.hash import pbkdf2_sha256;
#Hashes and returns password that is hashed and salted by 29000 rounds of pbkdf2 encryption
def hashPass(password):
    return pbkdf2_sha256.encrypt(password)

#Returns true if password hashes into hashpass, false otherwise
def verify(password, hashpass):
    return pbkdf2_sha256.verify(password,hashpass)

