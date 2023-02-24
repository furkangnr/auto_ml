import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fpdf import FPDF
import math
shap.initjs()

"""
TODO
----
    shap.plots içeriği araştırılacak
    konu bazında methodlar oluşturulacak (konuları belirle)
    shap.plots.bar(cluster) eklenecek
    shap_interaction eklenecek
    önemli değişkenler (shap > 1e-3) heatmap
    cohorts planlaması yapılacak
    __init__ içerisinde model_output ve feature_perturbation denemelerine bak
        - feature_perturbation = 'tree_path_dependent' default when data is None
        - feature_perturbation = 'interventional' default when data is not None
        - model_output = 'raw' (default of model like logit), 'probability', 'log_loss'

"""

class Shap:
    def __init__(self, model, X, y, cohorts = True): #, feature_perturbation = 'interventional', model_output = 'raw'):
        """
        This class helps to use some features of 'shap' package. With the help of this class;
            1) We can select the most important features for the defined model,
            2) We can get the reason of the output for any given observation,
            3) We can get the effect of features for the given model,
            4) We can check the quality of the data.

        Parameters
        ----------
            model : fitted model object for the data like xgboost.XGBClassifier.fit()
            X (pandas.DataFrame) : data that is used for operations. The data doesn't have to be the same data which is used for building the model.
            cohorts (bool) : Whether to create cohorts object (default) or not
            #feature_perturbation (str) : Way of computing the SHAP values.
            #     'interventional' approach uses the shared dataset to compute conditional expectations in the presence of correlated input features.
            #     'tree_path_dependent' approach, on the other hand, uses the training dataset and the number of samples that fall into each node.
            #model_output (str): unit of the SHAP values to explain the predictions from the leaf nodes of the trees.
            #    If it is 'raw', then it uses the unit of the model like logit as many of the models use as the output unit.

        Returns
        -------
            Shap object to use all methods.
        """
        params_type = {'X': [X, pd.core.frame.DataFrame]
                      ,'y': [y, pd.core.frame.Series]
                      ,'cohorts': [cohorts, bool]
                      #,'feature_perturbation': [feature_perturbation, str]
                      #,'model_output': [model_output, str]
                      }
        self.assert_type(params_type)
        params_choice = {'cohorts': [cohorts, [True, False]]
                        #,'feature_perturbation': [feature_perturbation, ['interventional', 'tree_path_dependent']]
                        #,'model_output': [model_output, ['raw', 'probability', 'log_loss']]
                        }
        self.assert_choice(params_choice)
        self.model = model
        #self.model._estimator_type -> 'classifier' or 'regressor'
        self.X = X.drop_duplicates()
        self.y = y
        #type(model).__name__ / model.__class__.__name__
        self.explainer = shap.TreeExplainer(self.model) #, self.X, feature_perturbation = feature_perturbation, model_output = model_output) #TO DO: Explainer type will be selected by the model type
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, list):
            self.expected_value = self.expected_value[-1]
        if cohorts:
            self.shap_values_cohorts = self.explainer(self.X)
            if len(self.shap_values_cohorts.shape) == 3:
                self.shap_values_cohorts = self.shap_values_cohorts[:,:,1]
            self.shap_values = self.shap_values_cohorts.values
            print('if', self.shap_values)
        else:
            self.shap_values = self.explainer.shap_values(self.X)
            if len(self.shap_values.shape) == 3:
                self.shap_values = self.shap_values[:,:,1]
            print('else', self.shap_values)
        
        self.shap_values_df = None
        self.shap_values_df_perc = None
        self.probas = None
        self.clustering = None
        self.shap_interaction_values = None
        #model type (multi output etc.) -> self.model.classes_
        #self.model._get_tags() ????
        #self.model.objective

    def assert_type(self, params):
        """
        This helper method returns error in case of an inappropriate type.

        Parameters
        ----------
            params (dict): parameter dictionary to check the type of a parameter.
                key -> name of the parameter
                value -> [type of given input, required type of given input]

        Returns
        -------
            Assert an error if given type is inappropriate. (AssertionError)
        """
        for key, value in params.items():
            assert isinstance(value[0], value[1]), "Given \"" + key + "\" parameter should be " + value[1].__name__ + " type, not " + type(value[0]).__name__  + "!"

    def assert_choice(self, params):
        """
        This helper method returns error in case of an inappropriate input.

        Parameters
        ----------
            params (dict): parameter dictionary to check the input of a parameter.
                key -> name of the parameter
                value -> [type of given input, [possible values for that parameter]]

        Returns
        -------
            Assert an error if given input is not one of the possible values. (AssertionError)
        """
        for key, value in params.items():
            assert value[0] in value[1], "Given \"" + key + "\" parameter should be one of " + str(value[1]) + " values, not " + str(value[0]) + "!"

    def replacing(self, x):
        replace_list = [['İ', 'I'], ['Ş', 'S'], ['Ç', 'C'], ['Ö', 'O'], ['Ü', 'U'], ['ı', 'i'], ['ü', 'u'], ['ç', 'c'], ['ş', 's'], ['ö', 'o'], ['Ğ', 'G'], ['ğ', 'g']]
        for r in replace_list:
            x = x.replace(r[0], r[1])
        return x
    
    def export(self, location = None, name = None, alt_name = None):
        """
        This helper method is an helper method to export a plot.

        Parameters
        ----------
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be the alt_name parameter.
            alt_name (str): the alternative name of the plot. It will be sent while calling the method.

        Returns
        -------
            Saves the plot.
        """
        location = location if location is not None else './'
        name = name.strip('/') if name is not None else alt_name
        if location[-1] != '/':
            location = location + '/'
        plt.savefig(location + name + '.png', dpi = 120, bbox_inches='tight')
        plt.close()

    def export_pdf(self, image_list, example_list, comment_list, headings, df_dictionary, location = None, name = 'explainability.pdf', title = 'EXPLAINABILITY REPORT', author = 'DEPARTMENT OF AI AND DATA SCIENCE', website = 'https://tr.linkedin.com/company/teb-arf?trk=public_profile_topcard-current-company'):
        """
        This helper method is used to create a pdf file that contains images.

        Parameters
        ----------
            image_list (list): name of the images that are exported.
            example_list (list): name of the images that are shown in the introduction part.
            comment_list (list): explanations of the images that are shown in the introduction part.
            location (str): the location to export the pdf file. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the pdf file which is being exported.

        Returns
        -------
            Exports all image into a pdf file.
        """
        class PDF(FPDF):
            def header(self):
                # font
                self.set_font('helvetica', 'B', 15)
                # Calculate width of title and position
                title_w = self.get_string_width(self.title) + 6
                doc_w = self.w
                self.set_x((doc_w - title_w) / 2)
                # colors of frame, background, and text
                self.set_draw_color(0, 80, 180) # border = blue
                self.set_fill_color(230, 230, 0) # background = yellow
                self.set_text_color(220, 50, 50) # text = red
                # Thickness of frame (border)
                self.set_line_width(1)
                # Title
                self.cell(title_w, 10, self.title, border=1, ln=1, align='C', fill=1)
                # Line break
                self.ln(10)

            # Page footer
            def footer(self):
                # Set position of the footer
                self.set_y(-15)
                # set font
                self.set_font('helvetica', 'I', 8)
               # Set font color grey
                self.set_text_color(169,169,169)
                # Page number
                self.cell(0, 10, f'Page {self.page_no()}', align='C')

            # Adding chapter title to start of each chapter
            def chapter_title(self, ch_title, link):
                # Set link location
                self.set_link(link)
                # set font
                self.set_font('helvetica', '', 12)
                # background color
                self.set_fill_color(200, 220, 255)
                # Chapter title
                chapter_title = f'{ch_title}'
                self.cell(0, 5, chapter_title, ln=1, fill=1)
                # line break
                self.ln()

            # Chapter content
            def chapter_body(self, image, txt):
                # Insert plot
                if image is not None:
                    pdf.image(image, x = pdf.w/8, w = pdf.w*3/4, h = pdf.h/2) # TODO: Resimleri ortalamaya dikkat et
                # set font
                self.set_font('times', '', 12)
                # insert text
                self.multi_cell(0, 5, txt)

            def print_table(self, header, table):
                # Effective page width, or just epw
                epw = pdf.w - 2*pdf.l_margin

                # evenly across table and page
                col_width1 = max(table.iloc[:, 0].apply(len).values + [len(table.columns[0])]) + 23
                col_width2 = epw - col_width1

                # Document title centered, 'B'old, 14 pt
                pdf.ln(3)
                pdf.set_font('times','B',14.0) 
                pdf.cell(epw, 0.0, header, align='C')
                pdf.set_font('times','B',10.0) 
                pdf.ln(3)

                # Text height is the same as current font size
                th = pdf.font_size
                pdf.cell(col_width1, 2*th, str(table.columns[0]), border=1)
                pdf.cell(col_width2, 2*th, str(table.columns[1]), border=1)

                pdf.set_font('times','',10.0) 
                pdf.ln(2*th)

                for row in range(0, len(table.index)):
                    # Enter data in colums
                    # Notice the use of the function str to coerce any input to the 
                    # string type. This is needed
                    # since pyFPDF expects a string, not a number.
                    y = 2*th if len(table.iloc[row, 1]) < 90 else 4*th
                    pdf.cell(col_width1, y, table.iloc[row, 0], border = 1)
                    pdf.multi_cell(col_width2, 2*th, table.iloc[row, 1], border=1)

            def print_chapter(self, ch_title, link, image, txt, table = None, table_header = None):
                self.add_page()
                self.chapter_title(ch_title, link)
                self.chapter_body(image, txt)
                if table is not None:
                    self.print_table(table_header, table)

            def print_image(self, ch_title, link, images):
                self.add_page()
                self.chapter_title(ch_title, link)
                for image in images:
                    pdf.image(image, x = pdf.w/8, w = pdf.w*3/4)
                    self.add_page()

        location = location if location is not None else './'
        if location[-1] != '/':
            location = location + '/'
        
        pdf = PDF('P', 'mm', 'Letter')
        pdf.set_title(title)
        pdf.set_author(author)

        # Create Links
        link_list = [website]
        for heading in headings:
            link_list.append(pdf.add_link())

        # Set auto page break
        pdf.set_auto_page_break(auto = True, margin = 15)

        # Add Page
        pdf.add_page()
        pdf.image('shap_header.png', x = -0.5, w = pdf.w + 1) 

        # Attach Links
        pdf.cell(0, 10, 'Website', ln = 1, link = link_list[0])
        for i in range(0, len(headings)):
            pdf.cell(0, 10, headings[i], ln = 1, link = link_list[i + 1])

        for i in range(0, len(headings)-1):
            table = None if headings[i] != 'Introduction' else df_dictionary
            table_header = None if headings[i] != 'Introduction' else 'Data Dictionary'
            pdf.print_chapter(headings[i], link_list[i + 1], example_list[i], comment_list[i], table = table, table_header = table_header)

        pdf.print_image(headings[-1], link_list[-1], image_list)
        pdf.output(location + name)

    def observation_3_indexes(self):
        """
        This helper method finds indexes of minimum, maximum and average predict probabilities in the data.

        Returns
        -------
            list of indexes
        """
        if self.probas is None:
            try:
                probas = self.model.predict_proba(self.X)[:,1]
            except:
                probas = self.model.predict(self.X)
            self.probas = pd.Series(probas, index=self.X.index)
        probas_sorted_list = self.probas.sort_values(ascending = True)
        probas_list = self.probas.tolist()
        a= probas_list.index(max(probas_list))
        b= probas_list.index(probas_sorted_list[int(len(probas_sorted_list)/2)])
        c= probas_list.index(min(probas_list))
        return [a, b, c]

    def shap2deltaprob(self,
                       features, 
                       shap_df, 
                       shap_sum,
                       probas,
                       func):
        '''
        An helper method to map shap to Δ probabilities
        Parameters
        ----------
            features (list): names of features
            shap_df (pd.DataFrame): dataframe containing shap values
            shap_sum (pd.Series): series containing shap sum for each observation
            probas (pd.Series): series containing predicted probability for each observation
            func_shap2probas (function): maps shap to probability (for example interpolation function)
        Returns
        -------
            pd.Series or pd.DataFrame of Δ probability for each shap value
        '''
        # 1 feature
        if type(features) == str or len(features) == 1:
            return probas - (shap_sum - shap_df[features]).apply(func)
        # more than 1 feature
        else:
            return shap_df[features].apply(lambda x: shap_sum - x).apply(func)\
                    .apply(lambda x: probas - x)

    def percentage(self):
        """
        This helper method calculates the percentage of shap values for all points.
        If sum of all shap values for all features except i, the contribution of feature i to the probability is the percentage for that feature

        Returns
        -------
            A dataframe of shap values which is converted to percantages.
        """
        try:
            probas = self.model.predict_proba(self.X)[:,1]
        except:
            probas = self.model.predict(self.X)        
        if self.probas is None:
            self.probas = pd.Series(probas, index=self.X.index)
        print(len(probas))  #####################
        print(len(self.shap_values)) 
        if self.shap_values_df is None:
            self.shap_values_df = pd.DataFrame(self.shap_values, columns = self.X.columns, index = self.X.index)
        
        print(self.shap_values_df.shape) #####################
        shap_sum = self.shap_values_df.sum(axis=1)
        shap_sum_sort = shap_sum.sort_values()
        print(self.probas)
        probas_sort = self.probas[shap_sum_sort.index]
        print(probas_sort)
        print(len(shap_sum_sort))
        print(len(probas_sort))       
        intp = interp1d(shap_sum_sort, probas_sort, bounds_error = False, fill_value="extrapolate")
        percentage_df = self.shap2deltaprob(self.X.columns.to_list(), self.shap_values_df, shap_sum, probas, intp)
        var_list = percentage_df.columns
        percentage_df['shap_sum'] = percentage_df.sum(axis=1,skipna=True).squeeze()
        base_value = np.full((len(probas),1), (math.exp(self.expected_value))/(1+math.exp(self.expected_value)))
        percentage_df['probas'] = probas
        percentage_df['base_value'] = base_value
        percentage_df['difference'] = (percentage_df['probas']-percentage_df['base_value'])/percentage_df['shap_sum']
        new_percentage = percentage_df[var_list].multiply(percentage_df.difference,axis=0)
        new_percentage['percentage_sum'] = new_percentage.sum(axis=1,skipna=True).squeeze()
        percentage_df = new_percentage.drop('percentage_sum',axis=1)
        percentage_df = percentage_df*100
        return percentage_df

    def calculate_interact(self):
        """
        TODO: doldurulacak
        Shapley values of interactions make it possible to quantify the impact of an interaction between two features on the prediction for each sample.
        """
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.X)
        return self.shap_interaction_values

    def documentation(self, df_dictionary, summary_bar = True, summary_dot = True, dependence = False, partial_dependence = False, waterfall = True, percentage = True, export = False, location = None, comment = False, name = 'explainability.pdf'):
        """
        This method is used for documentation process. We get a waterfall plot for an observation (default for the first one), 
        a bar plot of percentage contribution of features for that observation (default for the first one) and 
        a summary bar plot to show the feature importance of the model. With the help of 'dependence' and 'partial_dependence'
        parameters, we can show dependence and partial dependence plots of all features.

        Parameters
        ----------
            df_dict (pandas.DataFrame): Dictionary of the data.
            summary_bar (bool): Whether to show the summary bar plot of the model
            summary_dot (bool): Whether to show the summary dot plot of the model
            dependence (bool): Whether to show dependence plots for all features.
            partial_dependence (bool): Whether to show partial dependence plots for all features.
            waterfall (bool): Whether to show waterfall plots of three observations which have the minimum, maximum and median of the outcomes in the data.
            percentage (bool): Whether to show percentage bar plots of three observations which have the minimum, maximum and median of the outcomes in the data.
            export (bool): Whether to export all plots into a pdf file.
            location (str): the location to export the plots or the pdf. If it is None (default), then the location is the directory of the notebook.
            comment (bool): whether to export the explanations of plots.
            name (str): the name of the pdf file which is being exported.
        """
        params_type = {'df_dict': [df_dictionary, pd.DataFrame],
                       'dependence': [dependence, bool],
                       'partial_dependence': [partial_dependence, bool],
                       'summary_bar': [summary_bar, bool],
                       'summary_dot': [summary_dot, bool],
                       'waterfall': [waterfall, bool],
                       'percentage': [percentage, bool],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'comment': [comment, bool]
                      }
        self.assert_type(params_type)
        params_choice = {'summary_bar': [summary_bar, [True, False]],
                         'summary_dot': [summary_dot, [True, False]],
                         'dependence': [dependence, [True, False]],
                         'partial_dependence': [partial_dependence, [True, False]],
                         'waterfall': [waterfall, [True, False]],
                         'percentage': [percentage, [True, False]],
                         'export': [export, [True, False]],
                         'comment': [comment, [True, False]]
                        }
        self.assert_choice(params_choice)
        if export | comment:
            location = location if location is not None else './'
            if location[-1] != '/':
                location = location + '/'
        if waterfall | percentage | export:
            indexes = self.observation_3_indexes()
        if summary_bar | export:
            self.summary(plot_type = 'bar', max_display = None, export = True, location = location)
        if summary_dot | export:
            self.summary(plot_type = 'dot', max_display = None, export = True, location = location)
        if dependence | export:
            self.dependence_all(col2 = 'auto', export = True, location = location)
        if partial_dependence | export:
            self.partial_dependence_all(export = True, location = location)
        if waterfall | export:
            for i in indexes:
                self.feature_effect(i, plot_type = 'waterfall', export = True, location = location)
        if percentage | export:
            print('sample_plot_is_started')
            for i in indexes:
                self.sample_plot(i, plot_type = 'bar', percentage = True, export = True, location = location)
        lines = ['This report is prepared to give information about the explainability steps after building the model. Building the model is not the only proecss we need to consider. We need to know why the model takes the decision for that specific information. We will use some plots to understand how the model works. You can find some explanations about the process we take. After calling the method, we get a waterfall plot for an observation, a bar plot of percentage contribution of features for that observation and a summary bar plot to show the feature importance of the model. If the corresponding parameters are given as True, we also get dependence and partial dependence plots for all features. They are exported to the given location.',
                 'The summary bar plot shows the effect of features over the model. x-axis shows the sum of mean absolute shap values. The more the value, the more that feature has effect on the model. y-axis shows the names of the features. They are placed with respect to the sum of mean absolute shap values in ascending order. We can say that the most important features to predict the outcome are shown on the top of the y-axis and their magnitude are the width of the corresponding bar.',
                 'The summary dot plot shows the distribution of the features with this dot plot. Every point refers to a row in the data. y-axis shows the name of the features. x-axis shows the shap value of that row for that feature. If the color of the point is blue, it means that the value of that feature for that row is low with respect to other values for that feature in the data. If it goes form blue to red, it means that the value for that feature is increasing in the data. We can see some correlation with this dot plot. For example, if points are turning from blue to red through the x-axis, we can say that the output of the instance is increasing when the value of the feature is increasing or vice versa.',
                 'The dependence plot shows the scatter plot of the feature colored with the most correlated feature. x-axis shows the value of the feature, y-axis shows the shap value for that feature and the color shows the value of the most correlated feature. If the point is red or near to red, the value of the second feature is high and if it is blue, it is the opposite meaning of the red one. We can get some interaction information from the plot. We can see some points that are affected by the value of the second feature.',
                 'The partial dependence plot shows the partial dependence plot for that feature. x-axis shows the value of the feature and y-axis shows the expected value of the model given the value of the feature. The horizontal line shows the expected value of the model. We can observe the effect of the feature on the output of the model for that observation. If the blue line is under the horizontal line, it means that this feature is decreasing the output for that value.',
                 'The waterfall plot shows the effect of features for that observation. x-axis show the expected value of the model and y-axis shows features. Red bars show the positive effect and blue ones show the negative effect. Under the x-axis, we can see the expected value of the model without knowing any feature and at the top, we can see the expected value of the model after knowing the given information for that observation. This plot represents the outcome for only one observation. So, the values and colors of bars can change for another observation. This plot helps us to explain the outcome of the model for that observation. There will be three plots for this heading. First one is the observation that has the maximum outcome in the dataset. The second one is the observation that has the median of all outcomes. The last one is the observation that has the minimum outcome.',
                 'The bar plot shows the percentage effect of features for that observation. The only difference between bar plot and waterafll plot is the calculation of values. Blue color means that the feature decreases the probability of the outcome and red color is the opposite of the blue one. The values near to the bars mean that the corresponding feature increases/decreases the probability by that amount. Also, this plot is used for explaining only one observation after the prediction. There will be three plots for this heading. First one is the observation that has the maximum outcome in the dataset. The second one is the observation that has the median of all outcomes. The last one is the observation that has the minimum outcome.'
                ]
        if comment:
            with open(location + 'comments.txt', 'w') as f:
                f.write('\n'.join(lines[1:]))
        if export:
            lines = list(map(self.replacing, lines))
            df_dictionary = df_dictionary[df_dictionary['FEATURE_NAME'].isin(self.X.columns)]
            for i in range(len(df_dictionary.columns)):
                df_dictionary.iloc[:,i] = df_dictionary.iloc[:,i].apply(self.replacing)
            dependence_list = ['dependence_' + x + '_auto.png' for x in self.X.columns]
            partial_dependence_list = ['partial_dependence_' + x + '.png' for x in self.X.columns]
            waterfall_list = ['waterfall_' + str(x) + '.png' for x in indexes]
            percentage_list = ['sample_' + str(x) + '.png' for x in indexes]
            image_list = ['summary_bar_' + str(min(int(len(self.X.columns)), 30)) +'.png', 'summary_dot_' + str(min(int(len(self.X.columns)), 30)) +'.png'] + dependence_list + partial_dependence_list + waterfall_list + percentage_list
            example_list = [None, 'summary_bar_' + str(min(int(len(self.X.columns)), 30)) +'.png', 'summary_dot_' + str(min(int(len(self.X.columns)), 30)) +'.png', 'dependence_' + self.X.columns[0] + '_auto.png', 'partial_dependence_' + self.X.columns[0] + '.png', 'waterfall_' + str(indexes[0]) + '.png', 'sample_' + str(indexes[0]) + '.png']
            headings = ['Introduction', 'Summary Bar Plot', 'Summary Dot Plot', 'Dependence Plot (Colored)', 'Partial Dependence Plot', 'Waterfall Plot', 'Percentage Plot', 'All Plots']
            self.export_pdf(image_list, example_list, lines, headings, df_dictionary, location = location, name = name)

    def local_explainability(self, i, plot_type = 'waterfall'):
        """
        This method is used for local explainability stuff. We can get waterfall plot for the given observation and
        percentage bar plot of the observation.

        Parameters
        ----------
            i (int): Row number of the observation in the X to plot.
            plot_type (str): Plot type. It should be one of 'waterfall' (default), 'force' or 'decision'.

        """
        params_type = {'i': [i, int],
                       'plot_type': [plot_type, str],
                      }
        self.assert_type(params_type)
        params_choice = {'i': [i, np.arange(0, self.X.shape[0])],
                         'plot_type': [plot_type, ['force', 'waterfall', 'decision']]
                        }
        self.assert_choice(params_choice)
        self.feature_effect(i, plot_type = plot_type)
        self.sample_plot(i, plot_type = 'bar', percentage = True)

    def global_explainability(self, plot_type = 'violin', max_display = None, clustering_cutoff = 1):
        """
        This method is used to get plots about features to understand their effects in the model.

        Parameters
        ----------
            plot_type (str): Plot type. It should be one of 'waterfall' (default), 'force' or 'decision'.
            max_display (int): Number of features to plot (default None).
            clustering_cutoff (float): threshold value for shap.utils.hclust (default 1)

        """
        params_type = {'plot_type': [plot_type if plot_type is not None else "", str],
                       'ncol': [max_display if max_display is not None else 1, int],
                       'clustering_cutoff': [float(clustering_cutoff) if type(clustering_cutoff) == int else clustering_cutoff, float]
                      }
        self.assert_type(params_type)
        params_choice = {'plot_type': [plot_type, ['violin', 'bar', 'dot', 'compact_dot']],
                         'ncol': [max_display if max_display is not None else 1, np.arange(0, self.X.shape[1])]
                        }
        self.assert_choice(params_choice)
        self.summary(plot_type = plot_type, max_display = len(self.X.columns))
        self.cluster(clustering_cutoff = clustering_cutoff)

    def shap_df(self, percentage = False):
        """
        This method is used to get the shap values or percentage of shap values.

        Parameters
        ----------
            percentage (bool): whether to return the percentage of shap values or not (default).

        Returns
        -------
            Shap values (pandas.DataFrame)
        """
        params_type = {"percentage": [percentage, bool]}
        self.assert_type(params_type)
        params_choice = {"percentage": [percentage, [True, False]]}
        self.assert_choice(params_choice)
        if percentage & (self.shap_values_df_perc is None):
            self.shap_values_df_perc = self.percentage()
        if not percentage & (self.shap_values_df is None):
            self.shap_values_df = pd.DataFrame(self.shap_values, columns = self.X.columns, index = self.X.index)
        return self.shap_values_df_perc if percentage else self.shap_values_df

    def sample_plot(self, i, ncol = None, plot_type = 'pie', export = False, percentage = False, location = None, name = None, title = None, normalize = True):
        """
        TODO: Pie chart üzerine değer basma
        This method plots the shap values of the given input. This should be used especially for the percentage of shap values. 
        For other uses, "feature_effect" method gives better plots to understand.

        Parameters
        ----------
            i (int): Row number of the observation in the X to plot.
            ncol (int): Number of features to plot (default min(int(nfeature/2, 30))).
            plot_type (str): Plot type. It should be one of 'pie' (default) or 'bar'
            export (bool): Whether to export (default) the plot or not.
            percentage (bool): Whether to the percentage of shap values or not (default).
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'sample_' + str(i)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'sample_' + str(i)

        """
        try:
            probas = self.model.predict_proba(self.X)[:,1]
        except:
            probas = self.model.predict(self.X)      
        params_type = {'i': [i, int],
                       'ncol': [ncol if ncol is not None else 1, int],
                       'plot_type': [plot_type, str],
                       'export': [export, bool],
                       'percentage': [percentage, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'i': [i, np.arange(0, self.X.shape[0])],
                         'ncol': [ncol if ncol is not None else 1, np.arange(0, self.X.shape[1])],
                         'plot_type': [plot_type, ['pie', 'bar']],
                         'export': [export, [True, False]],
                         'percentage': [percentage, [True, False]]
                        }
        self.assert_choice(params_choice)
        if ncol is None:
            ncol = int(min(self.X.shape[1], 30))
        if percentage & (self.shap_values_df_perc is None):
            self.shap_values_df_perc = self.percentage()
        if not percentage & (self.shap_values_df is None):
            self.shap_values_df = pd.DataFrame(self.shap_values, columns = self.X.columns, index = self.X.index)
        temp = self.shap_values_df_perc.iloc[i:i+1,:] if percentage else self.shap_values_df.iloc[i:i+1,:]
        df = temp.transpose()
        df.columns = ['SHAP']
        df.index = [j + "(" + str(round(k[i], 3)) + ")" for j,k in zip(self.X.columns, self.X.values.T)]
        df['Colors'] = df.SHAP.apply(lambda x: 'red' if x > 0 else 'blue').values
        df['SHAP_ABS'] = df['SHAP'].abs()
        df.sort_values(by = 'SHAP_ABS', ascending = False, inplace = True)
        plt.figure(figsize = (12,12))
        if plot_type == 'pie':
            plt.pie(df.SHAP_ABS[:ncol], labels = df.index[:ncol], explode = [0.4]*ncol, colors = df.Colors[:ncol], radius=1.2, labeldistance = 1.1, rotatelabels = 270, normalize = normalize)
        elif plot_type == 'bar':
            plt.barh(df.index[:ncol], df.SHAP[:ncol], color = df.Colors[:ncol])
            max_ = df.SHAP_ABS[:ncol].max()
            for index, value in enumerate(df.SHAP[:ncol]):
                plt.text(value - 0.065*max_ if value < 0 else value + .01*max_, index - 0.25, str(round(value, 3)), color='black', fontweight='bold', fontsize=9)
        plt.tight_layout() # TODO: Kaldırıp dene
        alt_name = str(self.X.index[i]) + '  proba:' + str(round(probas[i],3)) + ' expected_proba:' + str(round((math.exp(self.expected_value))/(1+math.exp(self.expected_value)),3))
        alt_name2 = 'sample_' + str(i)
        plt.title(title if title is not None else alt_name)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)
        plt.show() 

    def feature_effect(self, i, plot_type = 'waterfall', link = 'identity', ncol = None, export = False, matplotlib = True, location = None, name = None, title = None):
        """
        This method plots the shap values of the given input. It is a better visualization of the shap values for the given observation.
        With these plots, we can understand the result of the model for that instance.
        The decision plot shows the values of the given input rather than the shap values.
        The force plot is good to see where the “output value” fits in relation to the “base value”.
        The water plot also allows us to see the amplitude and the nature of the impact of a feature with its quantification.
        The decision plot makes it possible to observe the amplitude of each change, “a trajectory” taken by a sample for the values of the displayed features.
        The decision plot, for a set of samples, quickly becomes cumbersome if we select too many samples. 
            It is very useful to observe a ‘trajectory deviation’ or ‘diverging/converging trajectories’ of a limited group of samples.

        Parameters
        ----------
            i (int): Row number of the observation in the X to plot.
            plot_type (str): Plot type. It should be one of 'waterfall' (default), 'force' or 'decision'.
            link (str): Type of values used in the plot. It should be one of 'identity' (default) or 'logit'.
            ncol (int): Number of features to plot (default min(int(nfeature/2, 30))).
            export (bool): Whether to export (default) the plot or not.
            matplotlib (bool): Whether to matplotlib backend (default) or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of plot_type + '_' + str(i)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of plot_type + '_' + str(i)

        """
        params_type = {'i': [i, int],
                       'plot_type': [plot_type, str],
                       'link': [link, str],
                       'ncol': [ncol if ncol is not None else 1, int],
                       'export': [export, bool],
                       'matplotlib': [matplotlib, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'i': [i, np.arange(0, self.X.shape[0])],
                         'plot_type': [plot_type, ['force', 'waterfall', 'decision']],
                         'link': [link, ['identity', 'logit']],
                         'ncol': [ncol if ncol is not None else 1, np.arange(0, self.X.shape[1])],
                         'export': [export, [True, False]],
                         'matplotlib': [matplotlib, [True, False]]
                        }
        self.assert_choice(params_choice)
        if ncol is None:
            ncol = int(min(self.X.shape[1], 30))
        alt_name = plot_type + '_' + str(i)
        plt.title(title if title is not None else alt_name)
        if plot_type == 'force':
            shap.force_plot(self.explainer.expected_value, self.shap_values[i,:], self.X.iloc[i,:], link = link, show = not export, matplotlib = matplotlib)
        elif plot_type == 'waterfall':
            shap.waterfall_plot(self.shap_values_cohorts[i], max_display = ncol, show = not export)
        elif plot_type == 'decision':
            shap.decision_plot(self.explainer.expected_value, self.shap_values[i], features = self.X, show = not export, link = link)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def dependence(self, col1, col2 = None, export = False, location = None, name = None, title = None):
        """
        This method gives the scatter plot of a feature. This plot can be colored with another feature.
        The dependency plot allows to analyze the features two by two by suggesting a possibility to observe the interactions. 
        The scatter plot represents a dependency between a feature(x) and the shapley values (y) colored by a second feature(hue).

        Parameters
        ----------
            col1 (str): Name of the feature to plot
            col2 (str): If it is None (default), it plots the scatter plot of the 'col1' feature. 
                        If it is 'auto', it finds the most correlated feature to color the scatter plot.
                        If it is one of the feature in the data, it colors the scatter plot with the given feature.
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'dependence_' + col1 + '_' + str(col2)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'dependence_' + col1 + '_' + str(col2)

        """
        params_type = {'col1': [col1, str],
                       'col2': [col2 if col2 is not None else "", str],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'col1': [col1, self.X.columns.tolist()],
                         'col2': [col2, ['auto', None] + self.X.columns.tolist()],
                         'export': [export, [True, False]]
                        }
        self.assert_choice(params_choice)
        alt_name = 'dependence_' + col1 + '_' + str(col2)
        title = title if title is not None else alt_name
        shap.dependence_plot(col1, self.shap_values, self.X, interaction_index = col2, show = not export, title = title)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def dependence_all(self, col2 = None, export = False, location = None, name = None, title = None):
        """
        TODO: Shap değeri 0 olan değişkenler için üretme düzenlemesi yapılabilir
        This method gives scatter plots of all features.

        Parameters
        ----------
            col2 (str): If it is None (default), it plots the scatter plot of the 'col1' feature. 
                        If it is 'auto', it finds the most correlated feature to color the scatter plot.
                        If it is one of the feature in the data, it colors the scatter plot with the given feature.
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'dependence_' + col1 + '_' + str(col2)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'dependence_' + col1 + '_' + str(col2)

        """
        for feature in self.X.columns:
            self.dependence(feature, col2 = col2, export = export, location = location, name = name, title = title)

    def summary(self, plot_type = None, max_display = None, export = False, location = None, name = None, title = None):
        """
        TODO: Açıklama yaz (nasıl yorumlanmalı)
        This method gives the summary plot of the features.
        The summary plot shows the most important features and the magnitude of their impact on the model.

        Parameters
        ----------
            plot_type (str): Plot type. It should be one of 'bar' (default), 'violin', 'dot' or 'compact_dot' (which is only used for SHAP interaction values). If it is None, it uses 'bar'
            max_display (int): Number of features to plot (default min(int(nfeature/2, 30))).
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'summary_' + str(plot_type) + '_' + str(max_display)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'summary_' + str(plot_type) + '_' + str(max_display)

        """
        params_type = {'plot_type': [plot_type if plot_type is not None else "", str],
                       'max_display': [max_display if max_display is not None else 1, int],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'plot_type': [plot_type, [None, 'violin', 'bar', 'dot', 'compact_dot']],
                         'max_display': [max_display if max_display is not None else 1, np.arange(0, self.X.shape[1])],
                         'export': [export, [True, False]]
                        }
        self.assert_choice(params_choice)
        if max_display is None:
            max_display = int(min(self.X.shape[1], 30))
        alt_name = 'summary_' + str(plot_type) + '_' + str(max_display)
        title = title if title is not None else alt_name
        shap.summary_plot(self.shap_values, self.X, plot_type = plot_type, max_display = max_display, show = not export, title = title)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def partial_dependence(self, col, export = False, location = None, name = None, title = None):
        """
        TODO: Açıklama yaz (Ne amaçla kullanılır, nasıl yorumlanır)
        This method gives the partial dependence plot of a feature.

        Parameters
        ----------
            col (str): Name of the feature to plot
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'partial_dependence_' + col
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'partial_dependence_' + col

        """
        params_type = {'col': [col, str],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'col': [col, self.X.columns.tolist()],
                         'export': [export, [True, False]]
                        }
        self.assert_choice(params_choice)
        shap.plots.partial_dependence(col, self.model.predict, self.X, ice = False, model_expected_value = True, feature_expected_value = True, show = not export)
        alt_name = 'partial_dependence_' + col
        plt.title(title if title is not None else alt_name)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def partial_dependence_all(self, export = False, location = None, name = None, title = None):
        """
        This method gives partial dependence plots of all features.

        Parameters
        ----------
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'dependence_' + col1 + '_' + str(col2)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'dependence_' + col1 + '_' + str(col2)

        """
        for feature in self.X.columns:
            self.partial_dependence(feature, export = export, location = location, name = name, title = title)

    def ABS_SHAP(self, ncol = None, export = False, location = None, name = None, title = None):
        """
        TODO: Açıklama yaz (Ne amaçla kullanılır)
        This method gives the colored bar plot of the summary plot.

        Parameters
        ----------
            ncol (int): Number of features to plot (default min(int(nfeature/2, 30)))
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'abs_shap_' + str(ncol)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'abs_shap_' + str(ncol)

        """
        params_type = {'ncol': [ncol if ncol is not None else 1, int],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'ncol': [ncol if ncol is not None else 1, np.arange(0, self.X.shape[1])],
                        'export': [export, [True, False]]
                        }
        self.assert_choice(params_choice)
        if ncol is None:
            ncol = int(min(self.X.shape[1], 30))
        shap_v = pd.DataFrame(self.shap_values)
        feature_list = self.X.columns
        shap_v.columns = feature_list

        corr_list = []
        for i in feature_list:
            b = np.corrcoef(shap_v[i], self.X[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis = 1).fillna(0)
        corr_df.columns = ['Variable', 'Corr']
        corr_df['Sign'] = np.where(corr_df['Corr']>0, 'red', 'blue')

        shap_abs = np.abs(shap_v)
        k = pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['Variable', 'SHAP_abs']
        k2 = k.merge(corr_df, left_on = 'Variable', right_on = 'Variable', how = 'inner')
        k2 = k2.sort_values(by = 'SHAP_abs').iloc[-ncol:, :]
        colorlist = k2['Sign']
        ax = k2.plot.barh(x= 'Variable', y = 'SHAP_abs', color = colorlist, figsize = (12, 10), legend = False)
        ax.set_xlabel('SHAP Value (Red = Positive Impact)')
        alt_name = 'abs_shap_' + str(ncol)
        plt.title(title if title is not None else alt_name)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def imp_features(self, ncol = 30):
        """
        TODO: Açıklama yaz (neye göre buluyor falan)
        This method gives the most important features in the model with respect to the given data.

        Parameters
        ----------
            ncol (int): Number of features to return (default 30)

        Returns
        -------
            Shap values of all features with feature names as index (pandas.DataFrame)
        """
        params_type = {'ncol': [ncol, int]}
        self.assert_type(params_type)
        params_choice = {'ncol': [ncol, np.arange(0, self.X.shape[1])]}
        self.assert_choice(params_choice)
        if self.shap_values_df is None:
            self.shap_values_df = pd.DataFrame(self.shap_values, columns = self.X.columns, index = self.X.index)
        shap_df = pd.DataFrame(self.shap_values_df.abs().mean(axis = 0).values, columns = ['shap'], index = self.X.columns)
        shap_features = shap_df.sort_values(by = 'shap', ascending = False)
        return shap_features.iloc[:ncol,:]

    def cluster(self, clustering_cutoff = 1, export = False, location = None, name = None, title = None):
        """
        TODO: Açıklama yaz

        Parameters
        ----------
            clustering_cutoff (float): threshold value for shap.utils.hclust.
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'decision_plot' + str(i)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'decision_plot' + str(i)

        """
        params_type = {'clustering_cutoff': [float(clustering_cutoff) if type(clustering_cutoff) == int else clustering_cutoff, float],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'export': [export, [True, False]]}
        self.assert_choice(params_choice)
        if self.clustering is None:
            self.clustering = shap.utils.hclust(self.X, self.y)

        alt_name = 'cluster'
        plt.title(title if title is not None else alt_name)
        shap.plots.bar(self.shap_values_cohorts, clustering=self.clustering, clustering_cutoff = clustering_cutoff)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def decision_plot(self, i, link='identity', export = False, location = None, name = None, title = None):
        """
        TODO: Açıklama yaz
        It is very useful to observe a ‘trajectory deviation’ or ‘diverging/converging trajectories’ of a limited group of samples.

        Parameters
        ----------
            i (int): Row number of the observation in the X to plot.
            link (str): Type of values used in the plot. It should be one of 'identity' (default) or 'logit'.
            export (bool): Whether to export (default) the plot or not.
            location (str): the location to export the plot. If it is None (default), then the location is the directory of the notebook.
            name (str): the name of the plot. If it is None (default), then the name will be in the form of 'decision_plot' + str(i)
            title (str): the title of the plot. If it is None (default), then the title will be in the form of 'decision_plot' + str(i)

        """
        params_type = {'i': [i, int],
                       'link': [link, str],
                       'export': [export, bool],
                       'location': [location if location is not None else "", str],
                       'name': [name if name is not None else "", str],
                       'title': [title if title is not None else "", str]
                      }
        self.assert_type(params_type)
        params_choice = {'i': [i, np.arange(0, self.X.shape[0])],
                         'link': [link, ['identity', 'logit']],
                         'export': [export, [True, False]]
                        }
        self.assert_choice(params_choice)
        alt_name = 'decision_plot_' + str(i)
        plt.title(title if title is not None else alt_name)
        shap.decision_plot(self.explainer.expected_value, self.shap_values[i], features = self.X, show = not export, link = link)
        if export:
            self.export(location = location, name = name, alt_name = alt_name)

    def decision_plot_multiple(self, ranges, export = False, location = None, name = None, title = None):
        """
        TODO: Kullanım alanlarını üret
        """
        pass
