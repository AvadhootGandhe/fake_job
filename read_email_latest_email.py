from simplegmail import Gmail
from simplegmail.query import construct_query

def read_latest_email():
    """
    Reads the latest email and returns its content in a format suitable for fraud detection.
    Returns a dictionary containing email details.
    """
    # Authenticate and connect
    gmail = Gmail()

    # You can specify additional query parameters if needed
    query_params = {
        "newer_than": (1, "day")  # optional filter to limit to recent emails
    }

    # Construct query (optional)
    query = construct_query(query_params)

    # Get inbox messages (default is most recent first)
    messages = gmail.get_messages(query=query)

    if not messages:
        raise Exception("No messages found.")

    message = messages[0]  # The latest email
    
    # Combine subject and body for text analysis
    email_content = f"{message.subject}\n{message.plain if message.plain else ''}"
    
    # Return email details
    return {
        'content': email_content,
        'sender': message.sender,
        'recipient': message.recipient,
        'subject': message.subject,
        'date': message.date,
        'snippet': message.snippet
    }

if __name__ == "__main__":
    try:
        email = read_latest_email()
        print("From:", email['sender'])
        print("To:", email['recipient'])
        print("Subject:", email['subject'])
        print("Date:", email['date'])
        print("Preview:", email['snippet'])
        print("\nContent preview:", email['content'][:500])
    except Exception as e:
        print(f"Error: {str(e)}")
