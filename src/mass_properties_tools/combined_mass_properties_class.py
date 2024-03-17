from typing import List


class CombinedMassProperties:
    def __init__(self, mp_list=None):
        if mp_list is None:
            mp_list = []
        self._mp_list = mp_list
        self._mp_calced = False
        self._mp = self.calc_combined_mass_props()
    
    @property
    def mp(self):
        return self.mp
    
    @property
    def mp_list(self):
        return self._mp_list
    
    def append_mass_properties(self, mp_list: List[MassProperties]):
        self._mp_list + mp_list
        self._mp_calced = False
        self._mp = self.calc_combined_mass_props()
    
    def calc_combined_mass_props(self):
        len_list = len(self._mp_list)
        self._mp_calced = True
        if len_list == 1:
            return self.mp_list[0]
        else:
            for i in range(0, len_list):
                mp_comb = combine_mass_props(
                    self.mp_list[i], 
                    self.mp_list[i+1])
            return mp_comb


def main() -> None:
    
    # metric mass unit: [kg]
    # metric inertia unit: [kg-m^2]
    pass
    

if __name__ == "__main__":
    main()
    