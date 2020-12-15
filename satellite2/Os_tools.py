#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:25:36 2020

@author: jordi
"""
import os

#class paths:
#    def __init__(self, path):
##        pass
#        self.path  = path
#    
#    def check(self):
#        if os.path.exists(self.path) == False:
#            os.mkdir(self.path)
#        if not self.path.endswith('/'):
#            self.path += '/'
#    def delete(self):
#        if os.path.exists(self.path) == False:
#            pass
#        else:
#            os.rmdir(self.path)
        
def check_path(path):
    if os.path.exists(path) == False:
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'

if __name__ == '__main__':        
    path = '/home/jordi/satellite/mix_shooting_dipole2/4'
    check_path(path)

#    paths('/home/jordi/satellite/mix_shooting_dipoli').check() 
#    paths('/home/jordi/satellite/mix_shooting_dipoli').delete()
