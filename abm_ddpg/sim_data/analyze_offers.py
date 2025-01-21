import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import numpy as np

def load_simulation_data(file_path: str) -> Dict:
    """Load simulation data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_offers_dataframe(simulation_data: Dict) -> pd.DataFrame:
    """Convert simulation data to pandas DataFrame for analysis"""
    rows = []
    market_prices = {}  # Her periyot için MCP'yi sakla
    
    for period, data in simulation_data['offers_by_period'].items():
        # MCP'yi kaydet
        market_prices[int(period)] = float(data['market_price'])
        
        # Teklifleri işle
        for agent_name, offers in data.items():
            if agent_name != 'market_price':  # market_price'ı atla
                for offer in offers:
                    rows.append({
                        'period': int(period),
                        'agent': offer['agent'],
                        'resource': offer['resource'],
                        'amount': offer['amount'],
                        'offer_price': float(offer['offerPrice']),
                        'acceptance_price': float(offer['acceptancePrice']),
                        'accepted': offer['acceptance'],
                        'budget': float(offer['budget'])
                    })
    
    df = pd.DataFrame(rows)
    df['market_price'] = df['period'].map(market_prices)  # MCP'yi DataFrame'e ekle
    return df

def save_statistics_to_json(df: pd.DataFrame, sim_id: str):
    """İstatistikleri JSON dosyasına kaydet"""
    stats = {
        'simulation_id': sim_id,
        'total_periods': len(df['period'].unique()),
        'agents': {}
    }
    
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        accepted_offers = len(agent_data[agent_data['accepted']])
        total_offers = len(agent_data)
        
        # Periyot bazlı bütçe değişimi
        budget_changes = []
        budget_evolution = agent_data.groupby('period')['budget'].last()
        for i in range(1, len(budget_evolution)):
            change = budget_evolution.iloc[i] - budget_evolution.iloc[i-1]
            budget_changes.append(change)
        
        stats['agents'][agent] = {
            'initial_budget': float(agent_data.iloc[0]['budget']),
            'final_budget': float(agent_data.iloc[-1]['budget']),
            'average_budget_change': float(np.mean(budget_changes)) if budget_changes else 0,
            'acceptance_rate': float(accepted_offers / total_offers * 100),
            'average_offer_price': float(agent_data['offer_price'].mean()),
            'average_accepted_price': float(agent_data[agent_data['accepted']]['offer_price'].mean()) if accepted_offers > 0 else 0,
            'resources': {}
        }
        
        # Kaynak bazlı istatistikler
        for resource in agent_data['resource'].unique():
            resource_data = agent_data[agent_data['resource'] == resource]
            resource_accepted = resource_data[resource_data['accepted']]
            
            stats['agents'][agent]['resources'][resource] = {
                'total_offers': len(resource_data),
                'accepted_offers': len(resource_accepted),
                'acceptance_rate': float(len(resource_accepted) / len(resource_data) * 100),
                'average_offer_price': float(resource_data['offer_price'].mean()),
                'average_accepted_price': float(resource_accepted['offer_price'].mean()) if len(resource_accepted) > 0 else 0
            }
    
    with open(f'simulation_{sim_id}_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

def analyze_simulation(file_path: str):
    """Analyze and visualize simulation data"""
    sim_data = load_simulation_data(file_path)
    if not sim_data['offers_by_period']:
        print(f"No offers found in {file_path}")
        return
    
    df = create_offers_dataframe(sim_data)
    sim_id = os.path.basename(file_path).split('_')[1]
    
    # 1. Tüm ajanların tekliflerini gösteren ana grafik
    plt.figure(figsize=(20, 16), dpi=100)
    
    # 1.1 All Agents Plot
    plt.subplot(2, 1, 1)
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        # Plot accepted offers with bigger markers
        accepted = agent_data[agent_data['accepted']]
        plt.scatter(accepted['period'], accepted['offer_price'], 
                   marker='o', label=f'{agent} (Accepted)', s=150)
        # Plot rejected offers with bigger markers
        rejected = agent_data[~agent_data['accepted']]
        plt.scatter(rejected['period'], rejected['offer_price'], 
                   marker='x', label=f'{agent} (Rejected)', s=150)
    
    # Plot market clearing price with thicker line
    periods = sorted(df['period'].unique())
    mcps = [df[df['period'] == period]['market_price'].iloc[0] for period in periods]
    plt.plot(periods, mcps, 
            'k--', label='Market Clearing Price (MCP)', linewidth=3)
    
    plt.title('All Agents: Offer Prices and MCP by Period', fontsize=16)
    plt.xlabel('Period', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 1.2 Budget Evolution (mevcut alt grafik)
    plt.subplot(2, 1, 2)
    
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        # Her periyottaki son bütçeyi al
        budget_evolution = agent_data.groupby('period')['budget'].last()
        
        plt.plot(budget_evolution.index, budget_evolution.values, 
                marker='o', linewidth=3, markersize=10,
                label=f'{agent} Budget')
    
    plt.title('Agent Budget Evolution Over Time', fontsize=16)
    plt.xlabel('Period', fontsize=14)
    plt.ylabel('Budget', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Alt grafikler arası boşluğu ayarla
    plt.subplots_adjust(hspace=0.3, right=0.85)
    
    plt.savefig(f'simulation_{sim_id}_all_agents_analysis.png', 
                bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    # 2. Her ajan için ayrı grafik (5 sütunlu genel bakış)
    num_agents = len(df['agent'].unique())
    num_cols = 5
    num_rows = (num_agents + num_cols - 1) // num_cols
    
    # Genel bakış grafiği
    fig_overview, axes_overview = plt.subplots(num_rows, num_cols, 
                                             figsize=(25, 5*num_rows), 
                                             dpi=100)
    if num_rows == 1:
        axes_overview = axes_overview.reshape(1, -1)
    
    # Renk haritası
    all_resources = df['resource'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_resources)))
    resource_colors = dict(zip(all_resources, colors))
    
    # Her ajan için hem genel bakışta hem de ayrı grafikte çiz
    for idx, agent in enumerate(df['agent'].unique()):
        agent_data = df[df['agent'] == agent]
        
        # Ayrı grafik için figür oluştur
        plt.figure(figsize=(15, 10), dpi=100)
        
        # Fonksiyon tekrarını önlemek için grafik çizme işlemini fonksiyona al
        def plot_agent_data(ax, is_individual=False):
            for resource in agent_data['resource'].unique():
                resource_data = agent_data[agent_data['resource'] == resource]
                color = resource_colors[resource]
                
                accepted = resource_data[resource_data['accepted']]
                if not accepted.empty:
                    ax.scatter(accepted['period'], accepted['offer_price'], 
                             marker='o', label=f'{resource} (Accepted)', 
                             s=150 if is_individual else 100, color=color)
                
                rejected = resource_data[~resource_data['accepted']]
                if not rejected.empty:
                    ax.scatter(rejected['period'], rejected['offer_price'], 
                             marker='x', label=f'{resource} (Rejected)', 
                             s=150 if is_individual else 100, color=color, alpha=0.5)
            
            ax.plot(periods, mcps, 'k--', label='MCP', linewidth=2)
            
            # İstatistikler
            accepted_rate = len(agent_data[agent_data['accepted']]) / len(agent_data) * 100
            stats_text = f'Acceptance: {accepted_rate:.1f}%'
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{agent}', fontsize=12 if is_individual else 10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if is_individual:
                ax.set_xlabel('Period', fontsize=12)
                ax.set_ylabel('Price', fontsize=12)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Genel bakış grafiğine çiz
        row, col = idx // num_cols, idx % num_cols
        plot_agent_data(axes_overview[row, col])
        
        # Ayrı grafiğe çiz
        plot_agent_data(plt.gca(), is_individual=True)
        plt.tight_layout()
        plt.savefig(f'simulation_{sim_id}_{agent}_analysis.png', 
                   bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()
    
    # Kullanılmayan alt grafikleri gizle
    for idx in range(num_agents, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes_overview[row, col].set_visible(False)
    
    plt.figure(fig_overview.number)
    plt.tight_layout()
    plt.savefig(f'simulation_{sim_id}_all_individual_analysis.png', 
                bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    # İstatistikleri JSON'a kaydet
    save_statistics_to_json(df, sim_id)

    # Print detailed statistics
    print(f"\nSimulation {sim_id} Detailed Analysis:")
    print("\nPeriod-by-Period Analysis:")
    for period in sorted(df['period'].unique()):
        print(f"\nPeriod {period}:")
        period_data = df[df['period'] == period]
        mcp = period_data['market_price'].iloc[0]
        print(f"Market Clearing Price (MCP): {mcp:.2f}")
        
        for agent in df['agent'].unique():
            agent_period_data = period_data[period_data['agent'] == agent]
            current_budget = agent_period_data['budget'].iloc[-1]
            
            print(f"\n  {agent}:")
            print(f"    Current Budget: {current_budget:.2f}")
            
            for _, offer in agent_period_data.iterrows():
                status = "ACCEPTED" if offer['accepted'] else "REJECTED"
                print(f"    {offer['resource']}: "
                      f"Offered {offer['offer_price']:.2f} -> {status} "
                      f"(Amount: {offer['amount']})")

    # Print summary statistics
    print("\nOverall Performance Summary:")
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        initial_budget = agent_data.iloc[0]['budget']
        final_budget = agent_data.iloc[-1]['budget']
        budget_change = final_budget - initial_budget
        accepted_offers = len(agent_data[agent_data['accepted']])
        total_offers = len(agent_data)
        acceptance_rate = (accepted_offers / total_offers) * 100
        
        print(f"\n{agent}:")
        print(f"  Initial Budget: {initial_budget:.2f}")
        print(f"  Final Budget: {final_budget:.2f}")
        print(f"  Total Budget Change: {budget_change:.2f}")
        print(f"  Acceptance Rate: {acceptance_rate:.1f}%")
        print(f"  Average Offer Price: {agent_data['offer_price'].mean():.2f}")

def main():
    # Analyze all simulation files in the directory
    for file in os.listdir('.'):
        if file.startswith('simulation_') and file.endswith('_offers.json'):
            print(f"\nAnalyzing {file}...")
            analyze_simulation(file)

if __name__ == "__main__":
    main() 