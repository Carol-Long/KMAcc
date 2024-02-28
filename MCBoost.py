'''
Functions for implementing multiaccuracy.
'''

'''
This code was produced by the authors of the paper "Universal adaptability: Target-independent inference that competes with propensity scoring"
Paper: https://www.pnas.org/doi/full/10.1073/pnas.2108097119
Code: https://osf.io/kfpr4/?view_only=adf843b070f54bde9f529f910944cd99
'''

import numpy as np
import numpy.ma as ma
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor


class ProbRange:
    """Simple wrapper class for a range of probabilities;
    lower bound is inclusive and upper bound is exclusive. """

    def __init__(self, lower_bound=float('-inf'), upper_bound=float('inf')):
        # We set our lower and upper bounds to -inf and +inf because
        # due to subtleties in how we boost probabilities, we may have
        # negative values or values > 1
        self.lower = lower_bound
        self.upper = upper_bound

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.lower == other.lower and self.upper == other.upper)

    def __ne__(self, other):
        return (not isinstance(other, self.__class__) or
                self.lower != other.lower or self.upper != other.upper)

    def __hash__(self):
        return (self.lower, self.upper).__hash__()


def within_range_mask(arr, prob_range):
    """
    Given a ProbRange and a numpy array of values, return a boolean mask
    of the values that are within the probability range.
    """
    return (arr >= prob_range.lower) & (arr < prob_range.upper)


def sigmoid(prob, center=False):
    """
    Applies sigmoid function to force probabilities to be between 0 and 1.
    If the probabilities start out being approximately in the range [0, 1],
    then the unaltered sigmoid will output probabilities all greater than 0.5,
    so we recenter the probs at 0.5
    """
    if center:
        prob -= 0.5
    return 1/(1 + np.exp(-1 * prob))


def rescale_prob(prob):
    """
    Rescales probabilities to be between zero and one.
    """
    pred_range = np.max(prob) - np.min(prob)
    new_preds = 1/(pred_range) * prob + (np.min(prob) / pred_range)
    return new_preds


def clip_prob(prob):
    """
    Clips probabilities to be between zero and one.
    """
    prob[prob < 0] = 0
    prob[prob > 1] = 1
    return prob


class RandomLinearPredictor:
    def __init__(self, random_seed=191):
        self.random_seed = random_seed

    def fit(self, data, labels):
        # labels are unused
        np.random.seed(self.random_seed)
        self.coefs = np.random.normal(size=data.shape[1])

    def predict(self, data):
        res = data @ self.coefs.T
        return sigmoid(res)


class ConstantPredictor:
    def __init__(self, constant=0.5):
        self.constant = constant

    def fit(self, data, labels):
        return

    def predict(self, data):
        return np.ones(len(data)) * 0.5


class ResidualFitter:
    def fit_to_resid(self, data, resid):
        try:
            return self.fit(data, resid)
        except Exception as detail:
            print("error:", detail)
        return None

    def fit(self, data, resid):
        raise Exception("Not Implemented")


class RidgeResidualFitter(ResidualFitter):
    def fit(self, data, resid):
        """
        Fits model(s) to predict the residual from the original data. Returns
        the correlation between the predictions and the residual and the model.
        """
        clf = Ridge(alpha=1)
        clf.fit(data, resid)
        h = clf.predict(data)  # TODO: should i use raw labels instead?
        corr = np.mean(h * resid)
        return corr, clf


class TreeResidualFitter(ResidualFitter):
    def fit(self, data, resid):
        """
        Fits trees(s) to predict the residual from the original data. Returns
        the correlation between the predictions and the residual and the model.
        """
        clf = DecisionTreeRegressor(max_depth=2)
        clf.fit(data, resid)
        h = clf.predict(data)
        corr = np.mean(h * resid)
        return corr, clf


class SubgroupModel:
    def __init__(self, subgroup_masks):
        # subgroup masks will allow us to fit the residual by subgroups
        self.subgroup_masks = subgroup_masks
        self.subgroup_preds = {}

    def fit(self, data, resid):
        for s, mask in enumerate(self.subgroup_masks):
            self.subgroup_preds[s] = np.mean(resid[mask])

    def predict(self, data, subgroup_masks=None):
        if subgroup_masks is None:
            for s, m in enumerate(self.subgroup_masks):
                if len(m) != len(data):
                    raise Exception("Length of data is incorrect")
        preds = np.zeros(len(data))
        for s, p in enumerate(self.subgroup_preds):
            preds[self.subgroup_masks[s]] = p
        return preds


class SubgroupFitter(ResidualFitter):
    def __init__(self, subgroup_masks):
        # subgroup masks will allow us to fit the residual by subgroups
        self.subgroup_masks = subgroup_masks

    def fit(self, data, resid):
        """
        Fits model(s) to predict the residual from the original data. Returns
        the correlation between the predictions and the residual and the model.
        """
        m = SubgroupModel(self.subgroup_masks)
        m.fit(data, resid)
        preds = m.predict(data)
        corr = np.mean(preds * resid)
        return corr, m


class SubpopPredictor:
    def __init__(self, subpop, value):
        self.subpop = subpop
        self.value = value

    def predict(self,data):
        return np.array([self.subpop(pt)*self.value for _,pt in data.iterrows()])


class SubpopFitter(ResidualFitter):
    def __init__(self, subpops):
        self.subpops = [(lambda name: lambda pt: pt[name])(attrName) for attrName in subpops]

    def fit(self, data, resid):
        worstCorr = 0
        worstSubpop = lambda pt: 0
        for S in self.subpops:
            sub = np.array([S(pt) for _,pt in data.iterrows()])
            corr = np.mean(sub * resid)
            #print(corr)
            if np.abs(corr) > np.abs(worstCorr):
                worstCorr = corr
                worstSubpop = S
        return abs(worstCorr),SubpopPredictor(worstSubpop,worstCorr)




class MCBoost:
    """
    Wrapper class for the multiaccuracy/multicalibration algorithm.
    """
    def __init__(self,
                 max_iter=5,
                 alpha=1e-4,
                 eta=1,
                 partition=False,
                 num_buckets=2,
                 bucket_strategy="simple",
                 rebucket=False,
                 multiplicative=False,
                 subpop_fitter=None,
                 subpops=None,
                 default_model_class=ConstantPredictor,
                 init_predictor=None
                 ):
        """
        Code to initialize an MCBoost object.
            - max_iter: The maximum number of iterations of the
                multicalibration/multiaccuracy method
            - alpha: accuracy parameter that determines the stopping condition
            - eta: parameter for multiplicative weight update (step size)
            - partition: boolean True/False flag for whether to split up
                predictions by their "partition" (e.g., predictions less than
                0.5 and predictions greater than 0.5)
            - num_buckets: how many buckets to split up the [0, 1]
                probability range
            - bucket_strategy: determines how the buckets will be "explored,"
                currently does nothing
            - rebucket: if True, we perform multicalibration; if false, we
                perform multiaccuracy
            - multiplicative: specifies the strategy for updating the weights
                (multiplicative weight vs additive)
            - subpops:  specifies a collection of characteristic attributes
            	and the values they take defining the S in subpops
                e.g. C = {'age': ['20-29','30-39','40+'], 'nJobs': [0,1,2,'3+'],... etc.}
            - subpop_fitter: specifies the type of model used to fit the
                residual fxn ('TreeResidualFitter' or 'RidgeResidualFitter' (default)).
            - random_seed: Mainly used for the default predictor
            - default_model_class: The class of the model that should be used
                as the MCBoost's default predictor model
            - init_predictor: the initial predictor function to use (i.e., if
                the user has a pretrained model)
        """
        # TODO(matthew): categorical & overlapping subgroups, specify a class
        # of functions over this data where we pass in an object that gives in
        # each of these functions list of functions, each function returns some
        # probability [is_male(), is_female(), is_black(), is_white()]
        # rather than predicting mean, evaluate residual and predict on this
        # subgroup
        self.max_iter = max_iter
        self.alpha = alpha
        self.eta = eta
        self.num_buckets = num_buckets
        self.bucket_strategy = bucket_strategy
        self.rebucket = rebucket
        self.partition = partition
        self.multiplicative = multiplicative
        self.iter_corrs = [0] * max_iter

        if subpops is not None:
            self.subpop_fitter = SubpopFitter(subpops)
      # elif subpop_fitter is not None:
      #     self.subpop_fitter = subpop_fitter() #not implemented
        elif subpop_fitter == 'TreeResidualFitter':
            self.subpop_fitter = TreeResidualFitter()
        elif subpop_fitter == 'RidgeResidualFitter':
            self.subpop_fitter = RidgeResidualFitter()
        else:
            self.subpop_fitter = RidgeResidualFitter()

        if init_predictor is None:
            dm = default_model_class()
            self.predictor = lambda x: dm.predict(x)
        else:
            self.predictor = init_predictor

        # for results of training process
        self.iter_models = []  # models fitted at each step
        self.iter_partitions = []  # keep track of the applicable partitions

    def multicalibrate(self, data, labels):
        """
        Performs multiaccuracy/multicalibration boost algorithm.
        (Multicalibration is achieved by setting "rebucket"=True)

        Given an initial hypothesis (in the form of the predictions
        on validation data), labels on validation data, an auditing
        algorithm, and an accuracy parameter alpha, returns a series of
        trained models that can be used to produce multiaccuracy-boosted
        predictions in combination with the original model.

        Returns a list of models and list of the applicable partitions.

        See paper https://arxiv.org/pdf/1805.12317.pdf (Kim et al. 2018).
        """
        pred_probs = self.predictor(data)
        resid = pred_probs - labels
        buckets = [ProbRange()]  # applies to all datapoints
        if self.partition and self.num_buckets > 1:
            frac = 1 / self.num_buckets
            buckets += [ProbRange(b * frac, (b+1) / frac)
                        for b in range(self.num_buckets)]
            buckets[-1].upper = 1.0  # deal with floating point rounding errors

        new_probs = np.array(pred_probs, copy=True)

        for it in range(self.max_iter):
            corrs = np.zeros(len(buckets))
            models = []

            # fit on various partitions
            probs = new_probs if self.rebucket else pred_probs
            for i, partition in enumerate(buckets):
                mask = within_range_mask(probs, partition)
                data_m = data[mask]
                resid_m = resid[mask]
                corrs[i], model = self.subpop_fitter.fit_to_resid(data_m,
                                                                  resid_m)
                models.append(model)


            if corrs.max() < self.alpha:  # lower than threshold
                for k in range(it, self.max_iter):
                    self.iter_corrs[k] = corrs[int(corrs.argmax())]
                break
            else:
                # update prediction probabilities
                self.iter_corrs[it] = corrs[int(corrs.argmax())]
                max_key = buckets[int(corrs.argmax())]
                prob_mask = within_range_mask(probs, max_key)
                self.iter_models.append(models[int(corrs.argmax())])
                self.iter_partitions.append(max_key)
                new_probs = self.update_probs(new_probs, self.iter_models[-1],
                                              data, mask=prob_mask)
                resid = new_probs - labels  # recalculate residuals

        return

    def update_probs(self, orig_preds, model, x, mask=None, **kwargs):
        """ Apply one multiplicative weight update.

        kwargs are passed to the predict function (check SubgroupFitter) """
        deltas = np.zeros(len(orig_preds))
        if mask is not None:  # only update the relevant probabilities
            deltas[mask] = model.predict(x[mask])
        else:
            deltas = model.predict(x, **kwargs)  # TODO: no kwargs hack
        # TODO: consider changing sigmoid to tanh function
        if self.multiplicative:
            update_weights = np.exp(-1 * self.eta * deltas)
            new_preds = update_weights * orig_preds  # sigmoid
        else:
            new_preds = orig_preds + deltas
        # new_preds = rescale_prob(new_preds)
        # new_preds = sigmoid(new_preds)
        new_preds = clip_prob(new_preds)
        return new_preds

    def predict_prob(self, x, t=float('inf'), **kwargs):
        # change method name to predict?
        # able to access predictions at various iterations
        """ Apply multiplicative weight updates using multiple models. Option
        to pass kwargs to the predict function via the update_probs function.
        """
        orig_preds = self.predictor(x)
        new_preds = np.array(orig_preds, copy=True)
        for i, m in enumerate(self.iter_models):
            if i <= t:
                probs = new_preds if self.rebucket else orig_preds
                mask = within_range_mask(probs, self.iter_partitions[i])
                new_preds = self.update_probs(new_preds, m, x, mask=mask,
                                              **kwargs)

        return new_preds

    def predict_all_prob(self, x, t=float('inf'), **kwargs):
        #return for all iterations up to t
        orig_preds = self.predictor(x)
        new_preds = np.array(orig_preds, copy=True)
        all_preds = [np.copy(new_preds)]
        for i, m in enumerate(self.iter_models):
            if i <= t:
                probs = new_preds if self.rebucket else orig_preds
                mask = within_range_mask(probs, self.iter_partitions[i])
                new_preds = self.update_probs(new_preds, m, x, mask=mask,
                                              **kwargs)
                all_preds.append(new_preds)
        for i in range(t - len(all_preds)):
            all_preds.append(new_preds)
        return all_preds

