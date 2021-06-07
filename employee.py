#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:41:21 2021

@author: halidziya
"""
class AbstractEmployee:
  def get_salary(self):
    raise NotImplementedError

class Employee(AbstractEmployee):
  def __init__(self, salary):
    self.salary = salary
  def get_salary(self):
    return self.salary