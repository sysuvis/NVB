# temp ly deprecated
import pandas as pd


class ID:
    def __init__(self, reg_no, ids):
        self.reg_no = reg_no
        self.ids = ids
        self.mapping = {}

    def add_mapping(self, to_no, map_to):
        self.mapping[to_no] = map_to


class IDManager:
    def __init__(self):
        self.__reg_ids = {}

    def register_id(self, reg_no, data=None):
        if reg_no not in self.__reg_ids:
            if data is None:
                self.__reg_ids[reg_no] = ID(reg_no, None)
            else:
                ids = data['idx'].values.tolist()
                self.__reg_ids[reg_no] = ID(reg_no, ids)

    def add_mapping(self, from_no, to_no, map_to):
        # number id
        self.__reg_ids[from_no].add_mapping(to_no, map_to)
        # print(self.__reg_ids[1].mapping)

    def map(self, from_no, to_no, ids):
        r = []
        mapping = self.__reg_ids[from_no].mapping.get(to_no, None)
        if mapping is None:
            return r
        for idx in ids:
            r += self.__reg_ids[from_no].mapping[to_no][idx]
        return r

    def get_all_reg_no(self):
        return self.__reg_ids.keys()

    def get_reg_ids(self):
        return self.__reg_ids

    def empty_r(self):
        r = {}
        for reg_no in self.__reg_ids:
            r[reg_no] = []
        return r

    def check_empty_r(self, r):
        for reg_no in self.__reg_ids:
            if len(r[reg_no]) > 0:
                return False
        return True
