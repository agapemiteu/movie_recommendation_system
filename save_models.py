"""
Save All Models and Data for Deployment
Run this after training all models in the notebook
"""

import os
import pickle
import sys

def save_all_models(model, tfidf_vectorizer, cosine_sim, nmf_model, 
                   user_features, movie_features, user_to_idx, movie_to_idx, 
                   idx_to_movie, user_id_map, movie_id_map, user_ids_dl, 
                   movie_ids_dl, movies_df, indices):
    """
    Save all trained models and data for deployment
    
    Parameters:
    -----------
    All trained models, mappings, and data from the notebook
    """
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("📦 SAVING ALL MODELS AND DATA FOR DEPLOYMENT")
    print("="*70)
    
    try:
        # 1. Save the Deep Learning Model
        print("\n[1/6] Saving Deep Learning model...")
        model.save(os.path.join(models_dir, 'deep_learning_model'))
        print("      ✓ Deep Learning model saved")
        
        # 2. Save TF-IDF model and cosine similarity matrix
        print("\n[2/6] Saving Content-Based models...")
        with open(os.path.join(models_dir, 'tfidf_model.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(os.path.join(models_dir, 'cosine_sim.pkl'), 'wb') as f:
            pickle.dump(cosine_sim, f)
        with open(os.path.join(models_dir, 'indices.pkl'), 'wb') as f:
            pickle.dump(indices, f)
        print("      ✓ TF-IDF vectorizer saved")
        print("      ✓ Cosine similarity matrix saved")
        print("      ✓ Movie indices saved")
        
        # 3. Save NMF Collaborative Filtering model
        print("\n[3/6] Saving Collaborative Filtering model...")
        with open(os.path.join(models_dir, 'nmf_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': nmf_model,
                'user_features': user_features,
                'movie_features': movie_features,
                'user_to_idx': user_to_idx,
                'movie_to_idx': movie_to_idx,
                'idx_to_movie': idx_to_movie
            }, f)
        print("      ✓ NMF model saved")
        print("      ✓ User and movie features saved")
        print("      ✓ ID mappings saved")
        
        # 4. Save Deep Learning ID mappings
        print("\n[4/6] Saving Deep Learning mappings...")
        with open(os.path.join(models_dir, 'dl_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'user_id_map': user_id_map,
                'movie_id_map': movie_id_map,
                'user_ids': user_ids_dl,
                'movie_ids': movie_ids_dl
            }, f)
        print("      ✓ User ID mappings saved")
        print("      ✓ Movie ID mappings saved")
        
        # 5. Save movies dataset
        print("\n[5/6] Saving movies dataset...")
        movies_df.to_csv(os.path.join(models_dir, 'movies.csv'), index=False)
        print("      ✓ Movies data saved")
        
        # 6. Verify all files
        print("\n[6/6] Verifying saved files...")
        expected_files = [
            'deep_learning_model',
            'tfidf_model.pkl',
            'cosine_sim.pkl',
            'indices.pkl',
            'nmf_model.pkl',
            'dl_mappings.pkl',
            'movies.csv'
        ]
        
        all_exist = True
        for file in expected_files:
            file_path = os.path.join(models_dir, file)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    print(f"      ✓ {file}/ (directory)")
                else:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"      ✓ {file} ({size_mb:.2f} MB)")
            else:
                print(f"      ✗ {file} NOT FOUND!")
                all_exist = False
        
        # Summary
        print("\n" + "="*70)
        if all_exist:
            print("🎉 ALL MODELS AND DATA SAVED SUCCESSFULLY!")
            print("="*70)
            print(f"\n📁 Location: {os.path.abspath(models_dir)}")
            print("\n✅ Ready for deployment!")
            print("   Run: streamlit run app.py")
            return True
        else:
            print("⚠️  WARNING: Some files are missing!")
            print("="*70)
            return False
            
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ ERROR: Failed to save models!")
        print(f"   {str(e)}")
        print("="*70)
        return False


if __name__ == "__main__":
    print("This script should be imported and called from the notebook.")
    print("It requires all trained models and data as parameters.")
