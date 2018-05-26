import sys
import os

from flask import Flask, request, session, g, redirect, url_for,\
        abort,render_template,flash,make_response,Markup,send_from_directory,send_file

reload(sys)
sys.setdefaultencoding('utf8')

Doc = Flask(__name__, static_url_path='/templates/html')
Doc.config.from_object(__name__)

@Doc.route('/')
def home(): 
    """
    doc index route
    """
    return render_template('html/index.html')

@Doc.route('/<path:filename>', methods=['GET', 'POST'])
def show_doxygen_file(filename): 
    """
    doc show local files
    """
    print filename
    return send_file( 'templates/html/' + filename)

