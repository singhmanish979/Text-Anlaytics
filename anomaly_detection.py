import json
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import scipy
from sklearn import preprocessing, decomposition, covariance
from matplotlib import pyplot as plt
import seaborn as sns
import re
from scipy.stats import chi2
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

from itertools import product
from collections import OrderedDict

def historical_data(data, dateField, lookback_window=pd.Timedelta(1,"Y")):
    """
    filter for only time window in comparison to wrangle_data
    """
    data=data.copy()
    data['start_time_utc'] = pd.to_datetime(data[dateField], utc=True, format="%Y-%m-%d %H:%M:%S")
    data = data[(data.start_time_utc >= pd.Timestamp("2018-08-01", tz="utc")) 
                & (data.start_time_utc >= pd.Timestamp.utcnow() - lookback_window)]
    data.set_index(data.start_time_utc, inplace=True)
    data['dummy'] = 1
    print(data.info())
    return data

def prepDict(indf, KEY, VALUE):
    """
    convert input dataframe rows to a dictionary of key, value pairs
    """
    di = dict(zip(indf[KEY], indf[VALUE]))
    return (di)


def e_cdf(df, col):
    n = len(col)
    x = np.sort(df[col])
    y = np.arange(1, len(x)+1) / len(x)
    return x, y


def cleanAndMap(indf, cleanVar, suff, *args):
    """
    clean text in the field and optionally
    group less frequent occurences in the field as MISC
    """
    freqTable = pd.DataFrame(indf[cleanVar].value_counts()).reset_index()
    freqTable.columns = [cleanVar, "count"]
    mappedVar = cleanVar + suff
    freqTable[[mappedVar]] = freqTable[[cleanVar]]
    if (len(args) > 0):
        freqTable.loc[freqTable["count"] < args[1], mappedVar] = args[0]
    # replace special characters with underscore
    freqTable[mappedVar] = freqTable[mappedVar].map(lambda x: re.sub(r'\W+', '_', x))
    # prepare a look up dictionary
    sol_dict = prepDict(freqTable, cleanVar, mappedVar)
    indf.loc[:, mappedVar] = indf[cleanVar].map(sol_dict)
    indf[mappedVar] = indf[mappedVar].str.lower()
    return (indf)


def wrangle_data(data, lookback_window=pd.Timedelta(1,"Y")):
    """
    """
    data=data.copy()
    data['start_time_utc'] = pd.to_datetime(data.start_time_utc, utc=True, format="%Y-%m-%d %H:%M:%S")
    data = data[(data.start_time_utc >= pd.Timestamp("2018-08-01", tz="utc")) 
                & (data.start_time_utc >= pd.Timestamp.utcnow() - lookback_window)]
    data.set_index(data.start_time_utc, inplace=True)
    data.loc[data.lob_desc == "Personal Notebooks", "lob_desc"] = "Inspiron Notebooks"
    data.loc[data.lob_desc == "Personal Desktops", "lob_desc"] = "Inspiron Desktops"
    data.loc[data.lob_desc == "Mobile Workstations", "lob_desc"] = "Precision Notebooks"
    data.loc[data.lob_desc == "Fixed Workstations", "lob_desc"] = "Precision Desktops"
    data.loc[data.lob_desc == "Chrome", "lob_desc"] = "Commercial Chrome"
    data['lob_desc'] = data.lob_desc.astype('category')
    data['dummy'] = 1
    print(data.info())
    return data

def category_counts_by_day(data, category, dummy_field="dummy"):
    """
    """
    daily_counts = data.groupby([data.index.date, category]).count()
    daily_counts = daily_counts.reset_index().rename(columns={"level_0":"date_"})
    daily_counts = daily_counts.pivot('date_', category, dummy_field).fillna(0)
    daily_counts.columns = daily_counts.columns.tolist()
    daily_counts["total"] = daily_counts.sum(axis=1)
    summed = daily_counts.sum()
    summed.name = "sum"
    daily_counts = daily_counts.append(summed)
    daily_counts.loc['sum','total'] = np.nan
    daily_counts.sort_values('sum', ascending=False, axis=1, inplace=True)
    return daily_counts.drop('sum')

def counts_to_fraction(counts):
    """
    """
    scaled = preprocessing.MaxAbsScaler().fit_transform(counts.T)
    return pd.DataFrame(scaled.T, index=counts.index, columns=counts.columns)

def calculate_distance_metrics(df, time_delta=pd.Timedelta(7, "D")):
    """
    """
    lobs = df.columns.tolist()
    # Self-join LoB fractions, offset by the time specified in the time_delta kwarg.
    df = df.merge(df.set_index(df.index.to_series() + time_delta), on='date_', suffixes=('','_prev'))

    # For each cumulative distribution fraction based metric, compare current to previous distribution:
    # Kolmogorov-Smirnov Distance
    df['ks_distance'] = df.apply(lambda row: scipy.stats.ks_2samp(row[lobs], row[[x+"_prev" for x in lobs]])[0], axis=1)
    #df['ks_distance_pvalue'] = df.apply(lambda row: scipy.stats.ks_2samp(row[lobs], row[[x+"_prev" for x in lobs]])[1], axis=1)
    # Wasserstein Distance (aka Earth-Mover)
    df["wasserstein"] = df.apply(lambda row: scipy.stats.wasserstein_distance(row[lobs], row[[x+"_prev" for x in lobs]]), axis=1)
    # Cramer-von Mises Distance (aka energy distance)
    df["energy_dist"] = df.apply(lambda row: scipy.stats.energy_distance(row[lobs], row[[x+"_prev" for x in lobs]]), axis=1)
    df.drop(columns=[x+"_prev" for x in lobs], inplace=True)
    
    # Calculate the Mahalanobis distance (based on the multivariate equivalent of the standard deviation)
    pca = decomposition.PCA().fit_transform(preprocessing.StandardScaler().fit_transform(df[lobs]))[:,:10]
    mahalanobis = covariance.MinCovDet().fit(pca).mahalanobis(pca)
    df['mahalanobis'] = mahalanobis
    return df.drop(columns=lobs)


def distance_metrics(df, time_delta=pd.Timedelta(7, "D")):
    """
    """
    lobs = df.columns.tolist()
    # Self-join LoB fractions, offset by the time specified in the time_delta kwarg.
    df = df.merge(df.set_index(df.index.to_series() + time_delta), on='date_', suffixes=('','_prev'))

    # For each cumulative distribution fraction based metric, compare current to previous distribution:
    # Kolmogorov-Smirnov Distance
    df['ks_distance'] = df.apply(lambda row: scipy.stats.ks_2samp(row[lobs], row[[x+"_prev" for x in lobs]])[0], axis=1)
    #df['ks_distance_pvalue'] = df.apply(lambda row: scipy.stats.ks_2samp(row[lobs], row[[x+"_prev" for x in lobs]])[1], axis=1)
    # Wasserstein Distance (aka Earth-Mover)
    df["wasserstein"] = df.apply(lambda row: scipy.stats.wasserstein_distance(row[lobs], row[[x+"_prev" for x in lobs]]), axis=1)
    # Cramer-von Mises Distance (aka energy distance)
    df["energy_dist"] = df.apply(lambda row: scipy.stats.energy_distance(row[lobs], row[[x+"_prev" for x in lobs]]), axis=1)
    df.drop(columns=[x+"_prev" for x in lobs], inplace=True)
    return df.drop(columns=lobs)


def find_anomalies(metric):
    """
    """
    m = metric.mean()
    std = metric.std()
    # A single point more than 3 standard deviations from the mean is an anomaly.
    threesigma = metric > m + std*3
    # Two out of the past three more than two sigma from the mean is too. 
    twosigma = metric.rolling(3).median() >  m + std*2
    # And four out of five more than one sigma from the mean.
    onesigma = metric.rolling(5).apply(lambda s: s.sort_values()[1], raw=False) >  m + std

    return threesigma|twosigma|onesigma

def send_email_notification(context, subFolder, *anomalies):
    """
    """
    sender = 'anomaly_detection@guidedresolution.dell.com'
    if anomalies:
        recipients = ['manish_singh20@dellteam.com','guided.resolution.dev@dell.com']
        if len(anomalies) == 1:
            subject = "WARNING: Anomaly detected in Guided Resolution Hits" + " for " + context
            msg = "Anomaly identified in: \n\n\t{}".format(anomalies[0])
        else:
            subject = "WARNING: Multiple Anomalies detected in Guided Resolution Hits" + " for " + context
            msg = "Anomalies identified in: \n\n\t{}".format("\n\t".join(anomalies))
        msg += "\n\nPlease investigate:\n\t"
        msg += "https://predictive.us.dell.com/u/Manish_Singh20/Factor-Anomaly-Detection-Deployed/browse/results"
    else:
        recipients = ['manish_singh20@dellteam.com','sharat_sharma@dell.com','randi_ludwig@Dell.com']
        subject = "SUCCESS: No Anomalies detected in Guided Resolution Hits" + " for " + context
        msg = "No further action necessary."
    # Create the root message and fill in the from, to, and subject headers
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = sender
    msgRoot['To'] = ", ".join(recipients)
    msgRoot.preamble = 'This is a multi-part message in MIME format.'
    # Encapsulate the plain and HTML versions of the message body in an
    # 'alternative' part, so message agents can decide which they want to display.
    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    msgText = MIMEText(msg)
    msgAlternative.attach(msgText)
    # We reference the image in the IMG SRC attribute by the ID we give it below
    if anomalies:
        html = "<h3>Anomaly identified in:</h3>" + "<br>    ".join(anomalies)
        html += "<br><br>Please investigate."
    else:
        html = "No further action necessary."
    html += "<br><br><a href='https://predictive.us.dell.com/u/Manish_Singh20/Factor-Anomaly-Detection-Deployed/browse/results'>Domino Job Results</a><br>"
    html += "<br>The charts below show the Kolmogorov-Smironv, Wasserstein, Cramér-von Mises, and Mahalanobis distances "
    html += "between the hits on Guided Resolution.<br>"
    html += "See <a href='https://www.idtools.com.au/detecting-outliers-using-mahalanobis-distance-pca-python/'>here</a> for "
    html += "an explanation of what the Mahalanobis distance is doing, and "
    html += "<a href='https://www.datadoghq.com/blog/engineering/robust-statistical-distances-for-machine-learning/'>here</a> for "
    html += "an even better explanation of the rest.<br>"
    html += "For Mahalanobis p-value refer <a href='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4825617/'>here</a> and "
    html += "<a href='https://pdfs.semanticscholar.org/582c/bd100ebae6888df2154b34afdc57469cdd20.pdf/'>here</a> for explanations on goodness-of-fit test for multivariate distributions of continuous data/sample.<br>"
    html += "<br>Included are day-over-day, week-over-week, and quarter-over-quarter comparisons. Anomalies are marked in red.<br>"
    html += "<br><br><h1>Day-over-Day:</h1>(Monday-Monday, Tuesday-Tuesday, etc.)<br><img src='cid:dailyproportionalVariateimage'><img src='cid:dailymultiVariateimage'>"
    html += "<br><br><h1>Week-over-Week:</h1><br><img src='cid:weeklyproportionalVariateimage'><img src='cid:weeklymultiVariateimage'>"
    html += "<br><br><h1>Quarter-over-Quarter:</h1><br><img src='cid:quarterlyproportionalVariateimage'><img src='cid:quarterlymultiVariateimage'><br>"
    msgText = MIMEText(html, 'html')
    msgAlternative.attach(msgText)
    # Open the images to be attached in binary format.
    for cchart in ['daily', 'weekly', 'quarterly']:
        for tchart in ['proportionalVariate', 'multiVariate']:
            with open("results/" + subFolder + "/" + "{}_{}_control_charts.png".format(cchart, tchart), 'rb') as png:
                msgImage = MIMEImage(png.read())
                # Define the image's ID as referenced above
                msgImage.add_header('Content-ID', '<{}{}image>'.format(cchart, tchart))
                msgRoot.attach(msgImage)
    # Send the email via smtp.
    s = smtplib.SMTP('ausxippc106.us.dell.com')
    s.sendmail(sender, recipients, msgRoot.as_string())
    s.quit()


def plot_counts_y( indf, *args ):
    for arg in args:
        f, ax = plt.subplots(figsize=(15,5))
        ax.plot(indf[arg])
        ax.set_ylabel(arg)
        plt.show()

        
def plot_counts_all( counts, lines_in_each_subplot):
    """
    """
    f,axes = plt.subplots(3,2,figsize=(15,20), sharex=True)
    axes = axes.flatten()
    ncols = len(counts.columns)
    for i, ax in enumerate(axes[:4]):
        counts.iloc[:-1, int(i*lines_in_each_subplot):int((1+i)*lines_in_each_subplot)].plot(ax=ax)
    counts.iloc[:-1, int((1+i)*lines_in_each_subplot):-1].plot(ax=axes[4])
    counts.iloc[:-1, -1:].plot(ax=axes[lines_in_each_subplot])
    for ax in axes:
        ax.set_ylabel("Hit Count")
        ax.set_xlabel("Date")
    f.autofmt_xdate()
    f.subplots_adjust(hspace=0.0, wspace=0.15)
    return f

    
def plot_counts(counts):
    """
    """
    f,axes = plt.subplots(3,2,figsize=(15,20), sharex=True)
    axes = axes.flatten()
    ncols = len(counts.columns)
    for i, ax in enumerate(axes[:4]):
        counts.iloc[:-1, int(i*6):int((1+i)*6)].plot(ax=ax)
    counts.iloc[:-1, int((1+i)*6):-1].plot(ax=axes[4])
    counts.iloc[:-1, -1:].plot(ax=axes[5])
    for ax in axes:
        ax.set_ylabel("Hit Count")
        ax.set_xlabel("Date")
    f.autofmt_xdate()
    f.subplots_adjust(hspace=0.0, wspace=0.15)
    return f

def plot_fractions(fractions):
    """
    """
    f,ax = plt.subplots(figsize=(12,12))
    fractions.plot(ax=ax, legend=False)
    ax.set_xlabel("Date")
    ax.set_ylabel("Hit Count Fraction")
    return f

def plot_cdf(fractions):
    """
    """
    with sns.color_palette("viridis", len(fractions)):
        f,ax = plt.subplots(figsize=(10,10))
        for dt in fractions.index[:-1]:
            ax.plot(fractions.loc[dt].cumsum().values, alpha=0.2)
        ax.grid(True)
        ax.set_xticklabels('')
        # ax.set_xticklabels(fractions.columns, rotation=45, horizontalalignment='right')
        # ax.xaxis.set_major_locator(plt.MaxNLocator(len(fractions.columns)))
        ax.set_ylabel("Cummulative Fraction of Guided Resolution Hits")
        plt.tight_layout()
        return f

def control_chart(metric, anomalies=None, ax=None, maha=False):
    """
    """
    if ax is None:
        f,ax = plt.subplots()
        
    if not maha:
        m = metric.mean()
        std = metric.std()
        if m - std*3 > metric.min():
            ax.axhline(m - std*3, c='0.85', ls=":")
        if m - std*2 > metric.min():
            ax.axhline(m - std*2, c='0.75', ls=":")
        if m - std > metric.min():
            ax.axhline(m - std, c='0.6', ls=":")
        ax.axhline(m, c='0.4', ls="--")
        ax.axhline(m + std, c='0.6', ls=":")
        ax.axhline(m + std*2, c='0.75', ls=":")
        ax.axhline(m + std*3, c='0.85', ls=":")
    
    ax.plot(metric.index, metric, '0.5')
    ax.plot(metric.index, metric, '.')
    if anomalies is None:
        ax.plot(metric[find_anomalies(metric)], 'r.')
    else:
        ax.plot(metric[anomalies], 'r.')
    
    ax.set_xlim(metric.index.min(), metric.index.max())
    ax.set_ylabel(metric.name)
    
def plot_control_chart(metrics, outfile=None):
    """
    """
    f, ax = plt.subplots(4,1,figsize=(12,15), sharex=True)
    control_chart(metrics.ks_distance, ax=ax[0])
    control_chart(metrics.wasserstein, ax=ax[1])
    control_chart(metrics.energy_dist, ax=ax[2])
    control_chart(metrics.mahalanobis, ax=ax[3])
    f.subplots_adjust(hspace=0.1)
    f.autofmt_xdate()
    if outfile is not None:
        f.savefig(outfile, bbox_inches="tight")
    return f

def control_charting(metrics, outfile=None):
    """
    """
    f, ax = plt.subplots(3,1,figsize=(12,15), sharex=True)
    control_chart(metrics.ks_distance, ax=ax[0])
    control_chart(metrics.wasserstein, ax=ax[1])
    control_chart(metrics.energy_dist, ax=ax[2])
    f.subplots_adjust(hspace=0.1)
    f.autofmt_xdate()
    if outfile is not None:
        f.savefig(outfile, bbox_inches="tight")
    return f


def plotVIF(indf):
    indf = indf[indf.Features != "Intercept"]
    f, ax = plt.subplots( figsize=(25,5) )
    indf.plot(ax=ax, kind='bar', legend=False) 
    ax.set_xlabel( "Features" )
    ax.set_ylabel( "Variance Inflation Factor (VIF)" )
    ax.set_xticklabels(indf.Features, rotation=40, ha='right')
    plt.axhline(y=3, c='0.75', ls='--')
    return f


def vif_visual(counts):
    # gather features
    sub_cols = [col for col in counts.columns if col not in ['total', 'MISC']]
    features = "+".join(sub_cols)

    counts_cp = counts.copy()
    counts_cp['dummmy_dep'] = 1
    # get y and X dataframes based on this regression:
    y, X = dmatrices('dummmy_dep ~' + features, counts_cp, return_type='dataframe')

    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Features"] = X.columns
    _ = plotVIF( vif )
    
def corr_plot(df):
    if 'MISC' in df.columns:
        df = df.drop(['MISC','total'], axis=1)
    else:
        df = df.drop(['total'], axis=1)
    f = plt.figure(figsize=(19, 19))
    plt.matshow(df.corr(), fignum=f.number, cmap=plt.cm.Blues)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    
    
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

        
def boxplot_outliers(indf, metric):
    f, ax = plt.subplots(1,1,figsize=(12,5), sharex=True)
    _ = ax.boxplot(indf[metric], sym='rs', vert=False, whis=1.5, labels=[metric], meanline=True)
    return f
   
    
def show_outliers(indf, metric):
    indf = indf.sort_values([metric], axis=0, ascending=True)
    q1, q3 = np.percentile(indf[metric], [25,75])
    iqr = (q3 - q1)
    lower_bound = q1 - (1.5 * iqr) 
    upper_bound = q3 + (1.5 * iqr)
    indf = indf[indf[metric + '_pvalue'] < 0.01]
    indf[metric + '_anomaly'] = 0
    indf.loc[(indf[metric] > lower_bound) & (indf[metric] > upper_bound), metric + '_anomaly'] = 1
    outdf = indf.loc[indf[ metric + '_anomaly']==1]
    return (outdf.sort_values('date_', axis=0, ascending=True))

def mah_control_chart(metrics, out_id , outfile=None):
    """
    """
    f, ax = plt.subplots(2,1, figsize=(12,10), sharex=True)
    control_chart(metrics.mahalanobis_pvalue, anomalies=out_id, ax=ax[0], maha=True)
    control_chart(metrics.mahalanobis, anomalies=out_id, ax=ax[1], maha=True)
    f.subplots_adjust(hspace=0.1)
    f.autofmt_xdate()
    if outfile is not None:
        f.savefig(outfile, bbox_inches="tight")
        
def replace_value(data, frequency=50, colname='brand_desc_mapped', into='misc_brand'):
    '''Written: Manish'''
    dt = pd.DataFrame(data[colname].value_counts())
    misc_brand = dt[dt[colname] <= frequency].index.tolist()
    print(' %s replace by %s'%(len(misc_brand),into))
    data.loc[data[colname].isin(misc_brand), colname] = into
    return data

def explode_row(df, var1='keywords', var2='start_time_utc'):
    '''Written: Manish'''
    var1_ = df[var1].str.strip().str.split(',', expand=True).values.ravel()
    var2_ = np.repeat(df.start_time_utc.values, len(var1_) / len(df))
    raw_df = pd.DataFrame({var2: var2_, var1: var1_})
    return raw_df[~raw_df[var1].isnull()].reset_index(drop=True)


def transform_row(data, index, col_to_split):
    '''Written: Manish'''
    data.reset_index(drop=True, inplace=True)
    _list = []
    for i in range(len(data)):
        try:
            a = OrderedDict({data[index][i]: data[col_to_split][i].split(',')})
            res = product(*a.values())
            for line in res:
                  _list.append([m+"="+n.strip() for m,n in zip(a.keys(), line)][0])
        except Exception as e:
            print(data[col_to_split][i], e)
    df = pd.Series(_list).str.split("=",expand=True)
    df.columns = [index, col_to_split]
    return df