from django.shortcuts import render
from .models import predict_rating_and_sentiment  

def review_create(request):
    if request.method == 'POST':
        review_text = request.POST.get('review_text')
        
        rating, sentiment = predict_rating_and_sentiment(review_text)
        
        context = {
            'review_text': review_text,
            'rating': rating,
            'sentiment': sentiment,
        }
        
        return render(request, 'reviews/review_result.html', context)
    
    return render(request, 'reviews/review_form.html')
