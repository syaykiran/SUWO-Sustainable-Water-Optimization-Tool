from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class NumberOfNondominatedSolutions(Column):
    def __init__(self, width=6, **kwargs):
        super().__init__("n_nds", width=width, **kwargs)

    def update(self, algorithm):
        if algorithm.opt is not None:
            self.value = len(algorithm.opt)


class MyOutput(Output):
    def __init__(self):
        super().__init__()

        self.igd = Column("igd")
        self.gd = Column("gd")
        self.hv = Column("hv")
        self.eps = Column("eps")
        self.n_nds = NumberOfNondominatedSolutions(width=7)
        self.eps_col = Column("eps", width=7)
        self.indicator = Column("indictr", width=7)
        self.empty_col = Column("", width=1)  # Empty column for spacing

        self.pen_min_sqtot = Column("P_MIN_SQ", width=8)
        self.pen_max_sqtot = Column("P_MAX_SQ", width=8)
        self.pen_end_sqtot = Column("P_END_SQ", width=8)
        self.pen_all_tot = Column("PEN_TOT", width=8)
        self.pen_sca = Column("PEN_SCAL", width=8)
        self.sto_dif = Column("STO_DIF", width=8)

        self.enr_raw = Column("ENR_RAW", width=8)
        self.enr_obj = Column("ENR_OBJ", width=8)
        self.irg_obj = Column("IRG_OBJ", width=8)
        self.eco_obj = Column("ECO_OBJ", width=8)

        self.eng_tot = Column("ENG_TOT", width=11)
        self.irg_tot = Column("IRG_TOT", width=11)
        self.reg_ratio = Column("AV_REG_RAT", width=11)

        self.columns += [self.n_nds,
                         self.eps_col,
                         self.indicator,
                         self.empty_col,

                         self.pen_min_sqtot,
                         self.pen_max_sqtot,
                         self.pen_end_sqtot,
                         self.pen_all_tot,
                         self.pen_sca,
                         self.empty_col,

                         self.enr_raw,
                         self.enr_obj,
                         self.irg_obj,
                         self.eco_obj,
                         self.empty_col,

                         self.eng_tot,
                         self.irg_tot,
                         self.reg_ratio,
                         self.empty_col,
                         self.sto_dif
                         ]

    def initialize(self, algorithm):
        self.pf = None
        self.indicator_no_pf = MultiObjectiveSpaceTermination()

    def update(self, algorithm):
        super().update(algorithm)

        for col in [self.igd, self.gd, self.hv, self.eps, self.indicator]:
            col.set(None)

        F, feas = algorithm.opt.get("F", "feas")
        F = F[feas]

        if len(F) > 0:
            if self.pf is not None:
                if feas.sum() > 0:
                    self.igd.set(IGD(self.pf, zero_to_one=True).do(F))
                    self.gd.set(GD(self.pf, zero_to_one=True).do(F))
                    if self.hv in self.columns:
                        self.hv.set(Hypervolume(pf=self.pf, zero_to_one=True).do(F))

            if self.indicator_no_pf is not None:
                ind = self.indicator_no_pf
                ind.update(algorithm)
                valid = ind.delta_ideal is not None

                if valid:
                    if ind.delta_ideal > ind.tol:
                        max_from = "ideal"
                        eps = ind.delta_ideal
                    elif ind.delta_nadir > ind.tol:
                        max_from = "nadir"
                        eps = ind.delta_nadir
                    else:
                        max_from = "f"
                        eps = ind.delta_f

                    self.eps_col.set(eps)
                    self.indicator.set(max_from)

        if algorithm.opt is not None:
            pen_min_sqtot = algorithm.opt.get("PEN_MIN_SQTOT")[0]
            pen_max_sqtot = algorithm.opt.get("PEN_MAX_SQTOT")[0]
            pen_end_sqtot = algorithm.opt.get("PEN_END_SQTOT")[0]
            pen_all_tot = algorithm.opt.get("PEN_ALL_TOTAL")[0]
            pen_sca = algorithm.opt.get("PEN_SCALE")[0]

            enr_raw = algorithm.opt.get("ENR_OBJ")[0] - algorithm.opt.get("PEN_SCALE")[0]
            enr_obj = algorithm.opt.get("ENR_OBJ")[0]
            irg_obj = algorithm.opt.get("IRG_OBJ")[0]
            eco_obj = algorithm.opt.get("ECO_OBJ")[0]

            eng_tot = algorithm.opt.get("ENG_TOT")[0]
            irg_tot = algorithm.opt.get("IRG_TOT")[0]
            reg_ratio = algorithm.opt.get("AVE_REG_RATIO")[0]

            sto_dif = algorithm.opt.get("STO_DIF")[0]

            self.pen_min_sqtot.set(pen_min_sqtot)
            self.pen_max_sqtot.set(pen_max_sqtot)
            self.pen_end_sqtot.set(pen_end_sqtot)
            self.pen_all_tot.set(pen_all_tot)
            self.pen_sca.set(pen_sca)

            self.enr_raw.set(enr_raw)
            self.enr_obj.set(enr_obj)
            self.irg_obj.set(irg_obj)
            self.eco_obj.set(eco_obj)

            self.eng_tot.set(eng_tot)
            self.irg_tot.set(irg_tot)
            self.reg_ratio.set(reg_ratio)
            self.sto_dif.set(sto_dif)
