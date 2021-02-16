#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 02:09:27 2021

@author: zankyo
"""

from google_drive_downloader import GoogleDriveDownloader as gdd

modelid = "1i9_Jm11asgI4hUO4GzuCYSL31DMe8j9i"
gdd.download_file_from_google_drive(file_id=modelid,
                                    dest_path="./model/model.zip",
                                    unzip=True,
                                    showsize=True)