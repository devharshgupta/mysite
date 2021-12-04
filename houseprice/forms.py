from django import forms


class HomePageForm(forms.Form):
    MY_CHOICES = (
        ("1", "Yes"),
        ("0", "No"),
    )

    bhk = forms.IntegerField(widget=forms.NumberInput(
        attrs={'class': 'form__input form__input--distance'}))
    lat = forms.FloatField(widget=forms.NumberInput(
        attrs={'class': 'form__input form__input--cadence', 'id': 'lat'}))
    long = forms.FloatField(widget=forms.NumberInput(
        attrs={'class': 'form__input form__input--cadence', 'id': "long"}))
    area = forms.FloatField(widget=forms.NumberInput(
        attrs={'class': 'form__input form__input--cadence'}))
    resale = forms.ChoiceField(choices=MY_CHOICES, widget=forms.Select(
        attrs={'class': 'form__input form__input--type'}))
    rera = forms.ChoiceField(choices=MY_CHOICES, widget=forms.Select(
        attrs={'class': 'form__input form__input--type'}))
    built = forms.ChoiceField(choices=MY_CHOICES, widget=forms.Select(
        attrs={'class': 'form__input form__input--type'}))
