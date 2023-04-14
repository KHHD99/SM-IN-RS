# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import logging
from scipy import sparse
from recommenders.utils.python_utils import (jaccard, lift, exponential_decay, get_top_k_scored_items, rescale)
from recommenders.utils import constants
# import the metric !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from recommenders.utils.python_utils import \
    ( HD_Jaccard, Dice, jaccard_3w, sokal_sneath_i, cosine, sorgenfrei, mountford, mcconnaughey, kulczynski_ii, 
    driver_kroeber, johnson, simpson, braun_banquet, fager_mcgowan, euclid, minkowski, lance_williams, hellinger, chord, 
    
    sokal_michener,  sokal_sneath_ii, sokal_sneath_iv,  sokal_sneath_v, pearson_i,  pearson_ii, pearson_iii,  pearson_heron_i, 
    pearson_heron_ii, baroni_urbani_buser_i, baroni_urbani_buser_ii, forbes_i, forbes_ii, yuleq, yuleq_w, tarantula, ample, 
    rogers_tanimoto, faith, gower_legendre, innerproduct, russell_rao, tarwid, dennis, gower, stiles, fossum, disperson, hamann,
    michael, peirce, eyraud, yuleq_d, mean_manhattan, vari, shapedifference, patterndifference
    )

# deplicated : czekanowski, nei_li, tanimoto, hamming, squared_euclid, canberra, manhattan, cityblock, bray_curtis,  ochiai_i, otsuka, 
# Make an alias to the metric !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

COOCCUR = "cooccurrence"
JACCARD = "jaccard"
LIFT = "lift"

HD_JACCARD = "HD_JACCARD" 
DICE = "DICE" 
JACCARD_3W = "JACCARD_3W" 
SOKAL_SNEATH_I = "SOKAL_SNEATH_I" 
COSINE = "COSINE" 
SORGENFREI = "SORGENFREI" 
MOUNTFORD = "MOUNTFORD" 
MCCONNAUGHEY = "MCCONNAUGHEY" 
KULCZYNSKI_II = "KULCZYNSKI_II" 
DRIVER_KROEBER = "DRIVER_KROEBER" 
JOHNSON = "JOHNSON" 
SIMPSON = "SIMPSON" 
BRAUN_BANQUET = "BRAUN_BANQUET" 
FAGER_MCGOWAN = "FAGER_MCGOWAN" 
EUCLID = "EUCLID" 
MINKOWSKI = "MINKOWSKI" 
LANCE_WILLIAMS = "LANCE_WILLIAMS"  
HELLINGER = "HELLINGER"  
CHORD = "CHORD"  
SOKAL_MICHENER = "SOKAL_MICHENER"  
SOKAL_SNEATH_II = "SOKAL_SNEATH_II"  
SOKAL_SNEATH_IV = "SOKAL_SNEATH_IV"  
SOKAL_SNEATH_V = "SOKAL_SNEATH_V"  
PEARSON_I = "PEARSON_I"  
PEARSON_II = "PEARSON_II"  
PEARSON_III = "PEARSON_III" 
PEARSON_HERON_I = "PEARSON_HERON_I"  
PEARSON_HERON_II = "PEARSON_HERON_II"  
BARONI_URBANI_BUSER_I = "BARONI_URBANI_BUSER_I"  
BARONI_URBANI_BUSER_II = "BARONI_URBANI_BUSER_II"  
FORBES_I = "FORBES_I"  
FORBES_II = "FORBES_II"  
YULEQ = "YULEQ"  
YULEQ_W = "YULEQ_W"  
TARANTULA = "TARANTULA"  
AMPLE = "AMPLE"  
ROGERS_TANIMOTO = "ROGERS_TANIMOTO" 
FAITH = "FAITH" 
GOWER_LEGENDRE = "GOWER_LEGENDRE" 
INNERPRODUCT = "INNERPRODUCT" 
RUSSELL_RAO = "RUSSELL_RAO" 
TARWID = "TARWID" 
DENNIS = "DENNIS" 
GOWER = "GOWER" 
STILES = "STILES" 
FOSSUM = "FOSSUM" 
DISPERSON = "DISPERSON" 
HAMANN = "HAMANN" 
MICHAEL = "MICHAEL" 
PEIRCE = "PEIRCE" 
EYRAUD = "EYRAUD" 
YULEQ_D = "YULEQ_D" 
MEAN_MANHATTAN = "MEAN_MANHATTAN" 
VARI = "VARI" 
SHAPEDIFFERENCE = "SHAPEDIFFERENCE" 
PATTERNDIFFERENCE = "PATTERNDIFFERENCE" 

logger = logging.getLogger()


class SARSingleNode:
    """Simple Algorithm for Recommendations (SAR) implementation

    SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history
    and items description. The core idea behind SAR is to recommend items like those that a user already has
    demonstrated an affinity to. It does this by 1) estimating the affinity of users for items, 2) estimating
    similarity across items, and then 3) combining the estimates to generate a set of recommendations for a given user.
    """

    def __init__(
            self,
            col_user=constants.DEFAULT_USER_COL,
            col_item=constants.DEFAULT_ITEM_COL,
            col_rating=constants.DEFAULT_RATING_COL,
            col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
            col_prediction=constants.DEFAULT_PREDICTION_COL,
            similarity_type=JACCARD,
            time_decay_coefficient=30,
            time_now=None,
            timedecay_formula=False,
            threshold=1,
            normalize=False,
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            similarity_type (str): ['cooccurrence', 'jaccard', 'lift'] option for computing item-item similarity
            time_decay_coefficient (float): number of days till ratings are decayed by 1/2
            time_now (int | None): current time for time decay calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this threshold will be removed
            normalize (bool): option for normalizing predictions to scale of original ratings
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! edit similarity_type  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if similarity_type not in [ COOCCUR, JACCARD, LIFT, HD_JACCARD, DICE, JACCARD_3W, SOKAL_SNEATH_I, COSINE, SORGENFREI,
                                    MOUNTFORD, MCCONNAUGHEY, KULCZYNSKI_II, DRIVER_KROEBER, JOHNSON, SIMPSON, BRAUN_BANQUET, 
                                    FAGER_MCGOWAN, EUCLID, MINKOWSKI, LANCE_WILLIAMS, HELLINGER, CHORD, 

                                    SOKAL_MICHENER, SOKAL_SNEATH_II, SOKAL_SNEATH_IV, SOKAL_SNEATH_V, PEARSON_I, PEARSON_II, PEARSON_III,
                                    PEARSON_HERON_I, PEARSON_HERON_II, BARONI_URBANI_BUSER_I, BARONI_URBANI_BUSER_II, FORBES_I, FORBES_II,
                                    YULEQ, YULEQ_W, TARANTULA, AMPLE, ROGERS_TANIMOTO, FAITH, GOWER_LEGENDRE, INNERPRODUCT, RUSSELL_RAO,
                                    TARWID, DENNIS, GOWER, STILES, FOSSUM, DISPERSON, HAMANN, MICHAEL, PEIRCE, EYRAUD, YULEQ_D, 
                                    MEAN_MANHATTAN, VARI, SHAPEDIFFERENCE, PATTERNDIFFERENCE]:
            raise ValueError(
                            ' Similarity type must be one of ["cooccurrence" | "jaccard" | "lift" | "HD_JACCARD" | "DICE" | "JACCARD_3W" | '
                            ' "SOKAL_SNEATH_I" | "COSINE" | "SORGENFREI" | "MOUNTFORD" | "MCCONNAUGHEY" | "KULCZYNSKI_II" | "DRIVER_KROEBER" | '
                            ' "JOHNSON" | "SIMPSON" | "BRAUN_BANQUET" | "FAGER_MCGOWAN" | "EUCLID" | "MINKOWSKI" | "LANCE_WILLIAMS" | '
                            ' "HELLINGER" | "CHORD" | "SOKAL_MICHENER" | "SOKAL_SNEATH_II" | "SOKAL_SNEATH_IV" | "SOKAL_SNEATH_V" | '
                            ' "PEARSON_I" | "PEARSON_II" | "PEARSON_III" | "PEARSON_HERON_I" | "PEARSON_HERON_II" | "BARONI_URBANI_BUSER_I" | '
                            ' "BARONI_URBANI_BUSER_II" | "FORBES_I" | "FORBES_II" | "YULEQ" | "YULEQ_W" | "TARANTULA" | "AMPLE" | '
                            ' "ROGERS_TANIMOTO" | "FAITH" | "GOWER_LEGENDRE" | "INNERPRODUCT" | "RUSSELL_RAO" | "TARWID" | "DENNIS" | "GOWER" | '
                            ' "STILES" | "FOSSUM" | "DISPERSON" | "HAMANN" | "MICHAEL" | "PEIRCE" | "EYRAUD" | "YULEQ_D" | "MEAN_MANHATTAN" | '
                            ' "VARI" | "SHAPEDIFFERENCE" | "PATTERNDIFFERENCE" | ]'
            )

        self.similarity_type = similarity_type
        self.time_decay_half_life = (
                time_decay_coefficient * 24 * 60 * 60
        )  # convert to seconds
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold
        self.user_affinity = None
        self.item_similarity = None
        self.item_frequencies = None

        # threshold - items below this number get set to zero in co-occurrence counts
        if self.threshold <= 0:
            raise ValueError("Threshold cannot be < 1")

        # set flag to capture unity-rating user-affinity matrix for scaling scores
        self.normalize = normalize
        self.col_unity_rating = "_unity_rating"
        self.unity_user_affinity = None

        # column for mapping user / item ids to internal indices
        self.col_item_id = "_indexed_items"
        self.col_user_id = "_indexed_users"

        # obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # The min and max of the rating scale, obtained from the training data.
        self.rating_min = None
        self.rating_max = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above map - map array index to actual string ID
        self.index2item = None

    def compute_affinity_matrix(self, df, rating_col):
        """Affinity matrix.

        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we're treating
        the ratings as the event weights.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.

        Args:
            df (pandas.DataFrame): Indexed df of users and items
            rating_col (str): Name of column to use for ratings

        Returns:
            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
        """

        return sparse.coo_matrix(
            (df[rating_col], (df[self.col_user_id], df[self.col_item_id])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

    def compute_time_decay(self, df, decay_column):
        """Compute time decay on provided column.

        Args:
            df (pandas.DataFrame): DataFrame of users and items
            decay_column (str): column to decay

        Returns:
            pandas.DataFrame: with column decayed
        """

        # if time_now is None use the latest time
        if self.time_now is None:
            self.time_now = df[self.col_timestamp].max()

        # apply time decay to each rating
        df[decay_column] *= exponential_decay(
            value=df[self.col_timestamp],
            max_val=self.time_now,
            half_life=self.time_decay_half_life,
        )

        # group time decayed ratings by user-item and take the sum as the user-item affinity
        return df.groupby([self.col_user, self.col_item]).sum().reset_index()

    def compute_cooccurrence_matrix(self, df):
        """Co-occurrence matrix.

        The co-occurrence matrix is defined as :math:`C = U^T * U`

        where U is the user_affinity matrix with 1's as values (instead of ratings).

        Args:
            df (pandas.DataFrame): DataFrame of users and items

        Returns:
            numpy.ndarray: Co-occurrence matrix
        """

        user_item_hits = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])),
            shape=(self.n_users, self.n_items),
        ).tocsr()
        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(item_cooccurrence >= self.threshold)

        return item_cooccurrence.astype(df[self.col_rating].dtype)

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage.

        Args:
            df (pandas.DataFrame): dataframe with user and item ids
        """

        # generate a map of continuous index values to items
        self.index2item = dict(enumerate(df[self.col_item].unique()))

        # invert the mapping from above
        self.item2index = {v: k for k, v in self.index2item.items()}

        # create mapping of users to continuous indices
        self.user2index = {x[1]: x[0] for x in enumerate(df[self.col_user].unique())}

        # set values for the total count of users and items
        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

    def fit(self, df):
        """Main fit method for SAR.

        .. note::

        Please make sure that `df` has no duplicates.

        Args:
            df (pandas.DataFrame): User item rating dataframe (without duplicates).
        """

        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        logger.info("Collecting user affinity matrix")
        if not np.issubdtype(df[self.col_rating].dtype, np.number):
            raise TypeError("Rating column data type must be numeric")

        # copy the DataFrame to avoid modification of the input
        select_columns = [self.col_user, self.col_item, self.col_rating]
        if self.time_decay_flag:
            select_columns += [self.col_timestamp]
        temp_df = df[select_columns].copy()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities")
            temp_df = self.compute_time_decay(df=temp_df, decay_column=self.col_rating)

        logger.info("Creating index columns")
        # add mapping of user and item ids to indices
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].apply(
            lambda item: self.item2index.get(item, np.NaN)
        )

        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].apply(
            lambda user: self.user2index.get(user, np.NaN)
        )

        if self.normalize:
            self.rating_min = temp_df[self.col_rating].min()
            self.rating_max = temp_df[self.col_rating].max()
            logger.info("Calculating normalization factors")
            temp_df[self.col_unity_rating] = 1.0
            if self.time_decay_flag:
                temp_df = self.compute_time_decay(
                    df=temp_df, decay_column=self.col_unity_rating
                )
            self.unity_user_affinity = self.compute_affinity_matrix(
                df=temp_df, rating_col=self.col_unity_rating
            )

        # affinity matrix
        logger.info("Building user affinity sparse matrix")
        self.user_affinity = self.compute_affinity_matrix(
            df=temp_df, rating_col=self.col_rating
        )

        # calculate item co-occurrence
        logger.info("Calculating item co-occurrence")
        item_cooccurrence = self.compute_cooccurrence_matrix(df=temp_df)

        # print("BinaryMat \n" ,BinaryMat)

        # free up some space
        del temp_df

        self.item_frequencies = item_cooccurrence.diagonal()

        logger.info("Calculating item similarity")
        if self.similarity_type == COOCCUR:
            logger.info("Using co-occurrence based similarity")
            self.item_similarity = item_cooccurrence.astype(
                df[self.col_rating].dtype)

        elif self.similarity_type == JACCARD:
            logger.info("Using jaccard based similarity")
            self.item_similarity = jaccard(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )

        elif self.similarity_type == LIFT:
            logger.info("Using lift based similarity")
            self.item_similarity = lift(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! call the newest similarity metric !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        elif self.similarity_type == HD_JACCARD:
            logger.info("Using HD_JACCARD based similarity")
            self.item_similarity = HD_Jaccard(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == DICE:
            logger.info("Using DICE based similarity")
            self.item_similarity = Dice(item_cooccurrence).astype(df[self.col_rating].dtype)
            
        elif self.similarity_type == JACCARD_3W:
            logger.info("Using JACCARD_3W based similarity")
            self.item_similarity = jaccard_3w(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SOKAL_SNEATH_I:
            logger.info("Using SOKAL_SNEATH_I based similarity")
            self.item_similarity = sokal_sneath_i(item_cooccurrence).astype(df[self.col_rating].dtype)
        
        elif self.similarity_type == COSINE:
            logger.info("Using COSINE based similarity")
            self.item_similarity = cosine(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SORGENFREI:
            logger.info("Using SORGENFREI based similarity")
            self.item_similarity = sorgenfrei(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == MOUNTFORD:
            logger.info("Using MOUNTFORD based similarity")
            self.item_similarity = mountford(item_cooccurrence).astype(df[self.col_rating].dtype)
            
        elif self.similarity_type == MCCONNAUGHEY:
            logger.info("Using MCCONNAUGHEY based similarity")
            self.item_similarity = mcconnaughey(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == KULCZYNSKI_II:
            logger.info("Using KULCZYNSKI_II based similarity")
            self.item_similarity = kulczynski_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == DRIVER_KROEBER:
            logger.info("Using DRIVER_KROEBER based similarity")
            self.item_similarity = driver_kroeber(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == JOHNSON:
            logger.info("Using JOHNSON based similarity")
            self.item_similarity = johnson(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SIMPSON:
            logger.info("Using SIMPSON based similarity")
            self.item_similarity = simpson(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == BRAUN_BANQUET:
            logger.info("Using BRAUN_BANQUET based similarity")
            self.item_similarity = braun_banquet(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == FAGER_MCGOWAN:
            logger.info("Using FAGER_MCGOWAN based similarity")
            self.item_similarity = fager_mcgowan(item_cooccurrence).astype(df[self.col_rating].dtype)
        
        elif self.similarity_type == EUCLID:
            logger.info("Using EUCLID based similarity")
            self.item_similarity = euclid(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == MINKOWSKI:
            logger.info("Using MINKOWSKI based similarity")
            self.item_similarity = minkowski(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == LANCE_WILLIAMS:
            logger.info("Using LANCE_WILLIAMS based similarity")
            self.item_similarity = lance_williams(item_cooccurrence).astype(df[self.col_rating].dtype)
            
        elif self.similarity_type == HELLINGER:
            logger.info("Using HELLINGER based similarity")
            self.item_similarity = hellinger(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == CHORD:
            logger.info("Using CHORD based similarity")
            self.item_similarity = chord(item_cooccurrence).astype(df[self.col_rating].dtype)

        ################### Similarity with parameter d ###################


        elif self.similarity_type == SOKAL_MICHENER:
            logger.info("Using SOKAL_MICHENER based similarity")
            self.item_similarity = sokal_michener(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == SOKAL_SNEATH_II:
            logger.info("Using SOKAL_SNEATH_II based similarity")
            self.item_similarity = sokal_sneath_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SOKAL_SNEATH_IV:
            logger.info("Using SOKAL_SNEATH_IV based similarity")
            self.item_similarity = sokal_sneath_iv(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SOKAL_SNEATH_V:
            logger.info("Using SOKAL_SNEATH_V based similarity")
            self.item_similarity = sokal_sneath_v(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PEARSON_I:
            logger.info("Using PEARSON_I based similarity")
            self.item_similarity = pearson_i(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PEARSON_II:
            logger.info("Using PEARSON_II based similarity")
            self.item_similarity = pearson_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PEARSON_III:
            logger.info("Using PEARSON_III based similarity")
            self.item_similarity = pearson_iii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PEARSON_HERON_I:
            logger.info("Using PEARSON_HERON_I based similarity")
            self.item_similarity = pearson_heron_i(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PEARSON_HERON_II:
            logger.info("Using PEARSON_HERON_II based similarity")
            self.item_similarity = pearson_heron_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == BARONI_URBANI_BUSER_I:
            logger.info("Using BARONI_URBANI_BUSER_I based similarity")
            self.item_similarity = baroni_urbani_buser_i(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == BARONI_URBANI_BUSER_II:
            logger.info("Using BARONI_URBANI_BUSER_II based similarity")
            self.item_similarity = baroni_urbani_buser_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == FORBES_I:
            logger.info("Using FORBES_I based similarity")
            self.item_similarity = forbes_i(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == FORBES_II:
            logger.info("Using FORBES_II based similarity")
            self.item_similarity = forbes_ii(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == YULEQ:
            logger.info("Using YULEQ based similarity")
            self.item_similarity = yuleq(item_cooccurrence).astype(df[self.col_rating].dtype)
        
        elif self.similarity_type == YULEQ_W:
            logger.info("Using YULEQ_W based similarity")
            self.item_similarity = yuleq_w(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == TARANTULA:
            logger.info("Using TARANTULA based similarity")
            self.item_similarity = tarantula(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == AMPLE:
            logger.info("Using AMPLE based similarity")
            self.item_similarity = ample(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == ROGERS_TANIMOTO:
            logger.info("Using ROGERS_TANIMOTO based similarity")
            self.item_similarity = rogers_tanimoto(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == FAITH:
            logger.info("Using FAITH based similarity")
            self.item_similarity = faith(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == GOWER_LEGENDRE:
            logger.info("Using GOWER_LEGENDRE based similarity")
            self.item_similarity = gower_legendre(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == INNERPRODUCT:
            logger.info("Using INNERPRODUCT based similarity")
            self.item_similarity = innerproduct(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == RUSSELL_RAO:
            logger.info("Using RUSSELL_RAO based similarity")
            self.item_similarity = russell_rao(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == TARWID:
            logger.info("Using TARWID based similarity")
            self.item_similarity = tarwid(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == DENNIS:
            logger.info("Using DENNIS based similarity")
            self.item_similarity = dennis(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == GOWER:
            logger.info("Using GOWER based similarity")
            self.item_similarity = gower(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == STILES:
            logger.info("Using STILES based similarity")
            self.item_similarity = stiles(item_cooccurrence).astype(df[self.col_rating].dtype)
       
        elif self.similarity_type == FOSSUM:
            logger.info("Using FOSSUM based similarity")
            self.item_similarity = fossum(item_cooccurrence).astype(df[self.col_rating].dtype)
    
        elif self.similarity_type == DISPERSON:
            logger.info("Using DISPERSON based similarity")
            self.item_similarity = disperson(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == HAMANN:
            logger.info("Using HAMANN based similarity")
            self.item_similarity = hamann(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == MICHAEL:
            logger.info("Using MICHAEL based similarity")
            self.item_similarity = michael(item_cooccurrence).astype(df[self.col_rating].dtype)
        
        elif self.similarity_type == PEIRCE:
            logger.info("Using PEIRCE based similarity")
            self.item_similarity = peirce(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == EYRAUD:
            logger.info("Using EYRAUD based similarity")
            self.item_similarity = eyraud(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == YULEQ_D:
            logger.info("Using YULEQ_D based similarity")
            self.item_similarity = yuleq_d(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == MEAN_MANHATTAN:
            logger.info("Using MEAN_MANHATTAN based similarity")
            self.item_similarity = mean_manhattan(item_cooccurrence).astype(df[self.col_rating].dtype)
            
        elif self.similarity_type == VARI:
            logger.info("Using VARI based similarity")
            self.item_similarity = vari(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == SHAPEDIFFERENCE:
            logger.info("Using SHAPEDIFFERENCE based similarity")
            self.item_similarity = shapedifference(item_cooccurrence).astype(df[self.col_rating].dtype)

        elif self.similarity_type == PATTERNDIFFERENCE:
            logger.info("Using PATTERNDIFFERENCE based similarity")
            self.item_similarity = patterndifference(item_cooccurrence).astype(df[self.col_rating].dtype)
        
        else:
            raise ValueError("Unknown similarity type: {}".format(self.similarity_type))

        # free up some space
        del item_cooccurrence

        logger.info("Done training")

    def score(self, test, remove_seen=False):
        """Score all items for test users.

        Args:
            test (pandas.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """

        # get user / item indices from test set
        user_ids = list(
            map(
                lambda user: self.user2index.get(user, np.NaN),
                test[self.col_user].unique(),
            )
        )
        if any(np.isnan(user_ids)):
            raise ValueError("SAR cannot score users that are not in the training set")

        # calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores")
        test_scores = self.user_affinity[user_ids, :].dot(self.item_similarity)

        # ensure we're working with a dense ndarray
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()

        if self.normalize:
            counts = self.unity_user_affinity[user_ids, :].dot(self.item_similarity)
            user_min_scores = (
                    np.tile(counts.min(axis=1)[:, np.newaxis], test_scores.shape[1])
                    * self.rating_min
            )
            user_max_scores = (
                    np.tile(counts.max(axis=1)[:, np.newaxis], test_scores.shape[1])
                    * self.rating_max
            )
            test_scores = rescale(
                test_scores,
                self.rating_min,
                self.rating_max,
                user_min_scores,
                user_max_scores,
            )

        # remove items in the train set so recommended items are always novel
        if remove_seen:
            logger.info("Removing seen items")
            test_scores += self.user_affinity[user_ids, :] * -np.inf

        return test_scores

    def get_popularity_based_topk(self, top_k=10, sort_top_k=True):
        """Get top K most frequently occurring items across all users.

        Args:
            top_k (int): number of top items to recommend.
            sort_top_k (bool): flag to sort top k results.

        Returns:
            pandas.DataFrame: top k most popular items.
        """

        test_scores = np.array([self.item_frequencies])

        logger.info("Getting top K")
        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        return pd.DataFrame(
            {
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

    def get_item_based_topk(self, items, top_k=10, sort_top_k=True):
        """Get top K similar items to provided seed items based on similarity metric defined.
        This method will take a set of items and use them to recommend the most similar items to that set
        based on the similarity matrix fit during training.
        This allows recommendations for cold-users (unseen during training), note - the model is not updated.

        The following options are possible based on information provided in the items input:
        1. Single user or seed of items: only item column (ratings are assumed to be 1)
        2. Single user or seed of items w/ ratings: item column and rating column
        3. Separate users or seeds of items: item and user column (user ids are only used to separate item sets)
        4. Separate users or seeds of items with ratings: item, user and rating columns provided

        Args:
            items (pandas.DataFrame): DataFrame with item, user (optional), and rating (optional) columns
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            pandas.DataFrame: sorted top k recommendation items
        """

        # convert item ids to indices
        item_ids = np.asarray(
            list(
                map(
                    lambda item: self.item2index.get(item, np.NaN),
                    items[self.col_item].values,
                )
            )
        )

        # if no ratings were provided assume they are all 1
        if self.col_rating in items.columns:
            ratings = items[self.col_rating]
        else:
            ratings = pd.Series(np.ones_like(item_ids))

        # create local map of user ids
        if self.col_user in items.columns:
            test_users = items[self.col_user]
            user2index = {x[1]: x[0] for x in enumerate(items[self.col_user].unique())}
            user_ids = test_users.map(user2index)
        else:
            # if no user column exists assume all entries are for a single user
            test_users = pd.Series(np.zeros_like(item_ids))
            user_ids = test_users
        n_users = user_ids.drop_duplicates().shape[0]

        # generate pseudo user affinity using seed items
        pseudo_affinity = sparse.coo_matrix(
            (ratings, (user_ids, item_ids)), shape=(n_users, self.n_items)
        ).tocsr()

        # calculate raw scores with a matrix multiplication
        test_scores = pseudo_affinity.dot(self.item_similarity)

        # remove items in the seed set so recommended items are novel
        test_scores[user_ids, item_ids] = -np.inf

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test_users.drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pandas.DataFrame): users to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pandas.DataFrame: top k recommendation items for each user
        """

        test_scores = self.score(test, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test[self.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set

        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test

        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """

        test_scores = self.score(test)
        user_ids = np.asarray(
            list(
                map(
                    lambda user: self.user2index.get(user, np.NaN),
                    test[self.col_user].values,
                )
            )
        )

        # create mapping of new items to zeros
        item_ids = np.asarray(
            list(
                map(
                    lambda item: self.item2index.get(item, np.NaN),
                    test[self.col_item].values,
                )
            )
        )
        nans = np.isnan(item_ids)
        if any(nans):
            logger.warning(
                "Items found in test not seen during training, new items will have score of 0"
            )
            test_scores = np.append(test_scores, np.zeros((self.n_users, 1)), axis=1)
            item_ids[nans] = self.n_items
            item_ids = item_ids.astype("int64")

        df = pd.DataFrame(
            {
                self.col_user: test[self.col_user].values,
                self.col_item: test[self.col_item].values,
                self.col_prediction: test_scores[user_ids, item_ids],
            }
        )
        return df
