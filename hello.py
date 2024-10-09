import secrets

# Generate a secure random secret key
secret_key = secrets.token_hex(16)  # Generates a 32-character hex string (128 bits)
print(secret_key)
