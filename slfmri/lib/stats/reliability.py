from ..errors import *


def test_reliability(data, subj_ids):
    '''
    '''
    from functools import partial
    from scipy.stats import pearsonr

    avr = data.mean(0)  # group average
    # testing whether the subject level components are correlated with its group average
    corr_to_avr = np.apply_along_axis(partial(pearsonr, y=avr), axis=1, arr=data)
    df = pd.DataFrame(dict(corr=corr_to_avr[:, 0], group=subj_ids))

    # Below code perform linear mixed model to estimate random effect for ICC analysis
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        md = smf.mixedlm("corr ~ 1", df, groups=df["group"])
        mdf = md.fit()

    # The result will only present how the correlation values are consistant across subject.
    return get_icc(mdf)


def get_icc(results):
    '''get the Intraclass Correlation Coefficient (ICC)'''
    icc = results.cov_re / (results.cov_re + results.scale)
    return icc.values[0, 0]


def lr_test(formula, data, groups):
    '''
    perform likelihood ratio test of random-effects

    # Examples
    icc = get_icc(mdf)
    lrt, p = lr_test("corr ~ 1", data=df, groups='group')

    print(f'ICC = {icc:.4f}')
    print(f'The LRT statistic: {lrt:.4f} (p = {p:.5})')
    '''
    # fit null model in mixed linear model
    from scipy.stats import chi2
    null_model = smf.mixedlm(formula, data=data, groups=groups) \
        .fit(reml=False)
    # fit OLS model
    ols_model = smf.ols(formula, data=data) \
        .fit()
    # get the LRT statistic and p-value
    lrt = np.abs(null_model.llf - ols_model.llf) * 2
    p = chi2.sf(lrt, 1)
    return (lrt, p)